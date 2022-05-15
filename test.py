# string = "ABOUT ABSOLUTELY ABUSE ACCESS ACCORDING ACCUSED ACROSS ACTION ACTUALLY AFFAIRS AFFECTED AFRICA AFTER AFTERNOON AGAIN AGAINST AGREE AGREEMENT AHEAD ALLEGATIONS ALLOW ALLOWED ALMOSTALREADY ALWAYS AMERICA  AMERICAN AMONG AMOUNT ANNOUNCED ANOTHER ANSWER ANYTHING AREAS AROUND ARRESTED ASKED ASKING ATTACK ATTACKS AUTHORITIES BANKS BECAUSE BECOME BEFORE BEHIND BEING BELIEVE BENEFIT BENEFITS BETTER BETWEEN BIGGEST BILLION BLACK BORDER BRING BRITAIN BRITISH BROUGHT BUDGET BUILD BUILDING BUSINESS BUSINESSES CALLED CAMERON CAMPAIGN CANCER CANNOT CAPITAL CASES CENTRAL CERTAINLY CHALLENGE CHANCE CHANGE CHANGES CHARGE CHARGES CHIEF CHILD CHILDREN CHINA CLAIMS CLEAR CLOSE CLOUD COMES COMING COMMUNITY COMPANIES COMPANY CONCERNS CONFERENCE CONFLICT CONSERVATIVE CONTINUE CONTROL COULD COUNCIL COUNTRIES COUNTRY COUPLE COURSE COURT CRIME CRISIS CURRENT CUSTOMERS DAVID DEATH DEBATE DECIDED DECISION DEFICIT DEGREES DESCRIBED DESPITE DETAILS DIFFERENCE DIFFERENT DIFFICULT DOING DURING EARLY EASTERN ECONOMIC ECONOMY EDITOR EDUCATION ELECTION EMERGENCY ENERGY ENGLAND ENOUGH EUROPE EUROPEAN EVENING EVENTS EVERY EVERYBODY EVERYONE EVERYTHING EVIDENCE EXACTLY EXAMPLE EXPECT EXPECTED EXTRA FACING FAMILIES FAMILY FIGHT FIGHTING FIGURES FINAL FINANCIAL FIRST FOCUS FOLLOWING FOOTBALL FORCE FORCES FOREIGN FORMER FORWARD FOUND FRANCE FRENCH FRIDAY FRONT FURTHER FUTURE GAMES GENERAL GEORGE GERMANY GETTING GIVEN GIVING GLOBAL GOING GOVERNMENT GREAT GREECE GROUND GROUP GROWING GROWTH GUILTY HAPPEN HAPPENED HAPPENING HAVING HEALTH HEARD HEART HEAVY HIGHER HISTORY HOMES HOSPITAL HOURS HOUSE HOUSING HUMAN HUNDREDS IMMIGRATION IMPACT IMPORTANT INCREASE INDEPENDENT INDUSTRY INFLATION INFORMATION INQUIRY INSIDE INTEREST INVESTMENT INVOLVED IRELAND ISLAMIC ISSUE ISSUES ITSELF JAMES JUDGE JUSTICE KILLED KNOWN LABOUR LARGE LATER LATEST LEADER LEADERS LEADERSHIP LEAST LEAVE LEGAL LEVEL LEVELS LIKELY LITTLE LIVES LIVING LOCAL LONDON LONGER LOOKING MAJOR MAJORITY MAKES MAKING MANCHESTER MARKET MASSIVE MATTER MAYBE MEANS MEASURES MEDIA MEDICAL MEETING MEMBER MEMBERS MESSAGE MIDDLE MIGHT MIGRANTS MILITARY MILLION MILLIONS MINISTER MINISTERS MINUTES MISSING MOMENT MONEY MONTH MONTHS MORNING MOVING MURDER NATIONAL NEEDS NEVER NIGHT NORTH NORTHERN NOTHING NUMBER NUMBERS OBAMA OFFICE OFFICERS OFFICIALS OFTEN OPERATION OPPOSITION ORDER OTHER OTHERS OUTSIDE PARENTS PARLIAMENT PARTIES PARTS PARTY PATIENTS PAYING PEOPLE PERHAPS PERIOD PERSON PERSONAL PHONE PLACE PLACES PLANS POINT POLICE POLICY POLITICAL POLITICIANS POLITICS POSITION POSSIBLE POTENTIAL POWER POWERS PRESIDENT PRESS PRESSURE PRETTY PRICE PRICES PRIME PRISON PRIVATE PROBABLY PROBLEM PROBLEMS PROCESS PROTECT PROVIDE PUBLIC QUESTION QUESTIONS QUITE RATES RATHER REALLY REASON RECENT RECORD REFERENDUM REMEMBER REPORTREPORTS RESPONSE RESULT RETURN RIGHT RIGHTS RULES RUNNING RUSSIA RUSSIAN SAYING SCHOOL SCHOOLS SCOTLAND SCOTTISH SECOND SECRETARY SECTOR SECURITY SEEMS SENIOR SENSE SERIES SERIOUS SERVICE SERVICES SEVEN SEVERAL SHORT SHOULD SIDES SIGNIFICANT SIMPLY SINCE SINGLE SITUATION SMALL SOCIAL SOCIETY SOMEONE SOMETHING SOUTH SOUTHERN SPEAKING SPECIAL SPEECH SPEND SPENDING SPENT STAFF STAGE STAND START STARTED STATE STATEMENT STATES STILL STORY STREET STRONG SUNDAY SUNSHINE SUPPORT SYRIA SYRIAN SYSTEM TAKEN TAKING TALKING TALKS TEMPERATURES TERMS THEIR THEMSELVES THERE THESE THING THINGS THINK THIRD THOSE THOUGHT THOUSANDS THREAT THREE THROUGH TIMES TODAY TOGETHER TOMORROW TONIGHT TOWARDS TRADE TRIAL TRUST TRYING UNDER UNDERSTAND UNION UNITED UNTIL USING VICTIMS VIOLENCE VOTERS WAITING WALES WANTED WANTS WARNING WATCHING WATER WEAPONS WEATHER WEEKEND WEEKS WELCOME WELFARE WESTERN WESTMINSTER WHERE WHETHER WHICH WHILE WHOLE WINDS WITHIN WITHOUT WOMEN WORDS WORKERS WORKING WORLD WORST WOULD WRONG YEARS YESTERDAY YOUNG"
# list = string.split()
# print(list)

# Check line 1-3 before deleting


import argparse
import json
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from lipreading.model import Lipreading
from preprocessing.transform import warp_img, cut_patch

STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]
START_IDX = 48
STOP_IDX = 68
# CROP_WIDTH = CROP_HEIGHT = 96
CROP_WIDTH = CROP_HEIGHT = 90

@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def load_model(config_path: Path):
    with config_path.open() as fp:
        config = json.load(fp)
    tcn_options = {
        'num_layers': config['tcn_num_layers'],
        'kernel_size': config['tcn_kernel_size'],
        'dropout': config['tcn_dropout'],
        'dwpw': config['tcn_dwpw'],
        'width_mult': config['tcn_width_mult'],
    }
    return Lipreading(
        num_classes=500,
        tcn_options=tcn_options,
        backbone_type=config['backbone_type'],
        relu_type=config['relu_type'],
        width_mult=config['width_mult'],
        extract_feats=False,
    )


def visualize_probs(vocab, probs, col_width=4, col_height=300):
    num_classes = len(probs)
    out = np.zeros((col_height, num_classes * col_width + (num_classes - 1), 3), dtype=np.uint8)
    for i, p in enumerate(probs):
        x = (col_width + 1) * i
        cv2.rectangle(out, (x, 0), (x + col_width - 1, round(p * col_height)), (255, 255, 255), 1)
    top = np.argmax(probs)
    cv2.addText(out, f'Prediction: {vocab[top]}', (10, out.shape[0] - 30), 'Arial', color=(255, 255, 255))
    cv2.addText(out, f'Confidence: {probs[top]:.3f}', (10, out.shape[0] - 10), 'Arial', color=(255, 255, 255))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=Path, default=Path('configs/lrw_resnet18_mstcn.json'))
    parser.add_argument('--model-path', type=Path, default=Path('models/lrw_resnet18_mstcn.pth.tar'))
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--queue-length', type=int, default=30)
    args = parser.parse_args()

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=args.device)
    model = load_model(args.config_path)
    model.load_state_dict(torch.load(Path(args.model_path), map_location=args.device)['model_state_dict'])
    model = model.to(args.device)

    mean_face_landmarks = np.load(Path('preprocessing/20words_mean_face.npy'))

    with Path('labels/500WordsSortedList.txt').open() as fp:
        vocab = fp.readlines()
    assert len(vocab) == 500

    queue = deque(maxlen=args.queue_length)

    with VideoCapture(0) as cap:
        while True:
            ret, image_np = cap.read()
            if not ret:
                break
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            all_landmarks = fa.get_landmarks(image_np)
            if all_landmarks:
                landmarks = all_landmarks[0]

                # BEGIN PROCESSING

                trans_frame, trans = warp_img(
                    landmarks[STABLE_PNTS_IDS, :], mean_face_landmarks[STABLE_PNTS_IDS, :], image_np, STD_SIZE)
                trans_landmarks = trans(landmarks)
                patch = cut_patch(
                    trans_frame, trans_landmarks[START_IDX:STOP_IDX], CROP_HEIGHT // 2, CROP_WIDTH // 2)

                cv2.imshow('patch', cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

                patch_torch = to_tensor(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)).to(args.device)
                queue.append(patch_torch)

                if len(queue) >= args.queue_length:
                    print("Predicting", end="--->")
                    with torch.no_grad():
                        model_input = torch.stack(list(queue), dim=1).unsqueeze(0)
                        logits = model(model_input, lengths=[args.queue_length])
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        probs = probs[0].detach().cuda().numpy()

                    print("Complete")
                    top = np.argmax(probs)
                    print(f'Prediction: {vocab[top]}', end="")
                    print(f'Confidence: {probs[top]}')

                    vis = visualize_probs(vocab, probs)
                    cv2.imshow('probs', vis)

                # END PROCESSING

                for x, y in landmarks:
                    cv2.circle(image_np, (int(x), int(y)), 2, (0, 0, 255))

            cv2.imshow('camera', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1)
            if key in {27, ord('q')}:  # 27 is Esc
                break
            elif key == ord(' '):
                cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
