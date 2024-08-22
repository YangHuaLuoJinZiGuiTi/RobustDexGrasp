import os
import random
import sys
import subprocess
import glob
import cv2
import time

RANDOM_ORDER = False
FILENAME = 'good_video_samples.txt'
OPTIONS = {
    'y': 'yes',
    'n': 'no',
    'r': 'replay',
    'g': 'skip to next grasp',
    'o': 'skip to next object',
}


def opencv_play_video(video_f):
    cap = cv2.VideoCapture(video_f)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    n_frame = 0
    while (cap.isOpened()):
        ret, frame = cap.read()



        if ret == True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            elif n_frame>200:
                break
        else:
            break
        if n_frame == 0:
            time.sleep(2)

        n_frame+=1

    cap.release()
    cv2.destroyAllWindows()


def save_video_name(video_f):
    if os.path.exists(FILENAME):
        with open(FILENAME, 'a') as f:
            f.write(video_f + '\n')
    else:
        with open(FILENAME, 'w') as f:
            f.write(video_f + '\n')


if __name__ == '__main__':
    render_folder = sys.argv[1]
    all_video_files = []

    if os.path.exists(FILENAME):
        inp = input(f'There is already a file named {FILENAME}\nDo you want to overwrite? [y/(n)]')
        if inp != 'y':
            exit('Exiting...')

    print('Listing files...')
    for filename in glob.iglob(render_folder + '**/**/*.mp4', recursive=True):
        if filename.endswith('snow.mp4'):
            continue
        all_video_files.append(filename)

    print(f'Number of result videos: {len(all_video_files)}')
    all_video_files = sorted(all_video_files)

    # import IPython; IPython.embed(); exit()

    if RANDOM_ORDER:
        random.shuffle(all_video_files)

    reply = 'r'
    prev_grasp = None
    prev_obj = None
    for idx, video_f in enumerate(all_video_files):
        if idx % 3 ==0:
            if idx == 0:
                print('Here is your options:')
                for k,v in OPTIONS.items():
                    print('['+k+']', v)

            if reply == 'g':
                if prev_grasp == video_f.split('/')[-3]:
                    continue
                else:
                    print(f'Skipping to next grasp {video_f.split("/")[-3]} (previous: {prev_grasp})')

            if reply == 'o':
                if prev_obj == video_f.split('/')[-4]:
                    continue
                else:
                    print(f'Skipping to next object {video_f.split("/")[-4]} (previous: {prev_obj})')

            print(f'{idx}/{len(all_video_files)}')

            reply = 'r'
            while reply == 'r':
                # print(f'Opening video {video_f}')
                # subprocess.call(['xdg-open', video_f])
                opencv_play_video(video_f)

                reply = input('Is this a good one? [y/r/g/o/(n)]')
                print(f'Your answer {reply}')

                if reply == 'y':
                    save_video_name(video_f)

            prev_obj = video_f.split('/')[-4]
            prev_grasp = video_f.split('/')[-3]
            print(prev_obj, prev_grasp)