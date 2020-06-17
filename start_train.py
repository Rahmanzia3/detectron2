import numpy as np
import subprocess
import telegram
from telegram.ext import Updater
import datetime
from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import time


def send_tele(data,image_path):
    updater = Updater('805281461:AAH09xnakEe8MxOBLQ7jWaiNolGxoZyxxrM')
    dp = updater.dispatcher
    token='805281461:AAH09xnakEe8MxOBLQ7jWaiNolGxoZyxxrM'
    bot = telegram.Bot(token=token)
    # if name=="unknown":
    bot.send_photo(chat_id='-496362146', photo=open(image_path,'rb'))
    bot.send_message(chat_id='-496362146', text=data+ str(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))



def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines
def start_plot():
    experiment_folder = './output'
    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

    plt.plot(
        [x['iteration'] for x in experiment_metrics], 
        [x['total_loss'] for x in experiment_metrics])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'total_val_loss' in x], 
        [x['total_val_loss'] for x in experiment_metrics if 'total_val_loss' in x])
    plt.legend(['total_loss', 'total_val_loss'], loc='upper left')

    plt.savefig('Plot.png')







if __name__ == "__main__":
    # args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    # # exit()496362146

    parser = ArgumentParser()

    parser.add_argument("--data_source", default= "datasets/licenseplates", help="Where all custom data is stored or a download link")


    parser.add_argument("--batch", default=3, help="Batch size")
    # parser.add_argument("--image_type", default= 'jpg,png,jpge' , help="Image extentions(split by comma)")
    parser.add_argument("--iterations", default=40, help="Number of Iterations to run" , type = int)
    parser.add_argument("--num_gpus", default=1, help="NUmber of GPus" , type = int)

    parser.add_argument("--cfg_model", default='COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml', help="CFG model")
    # parser.add_argument("--download_url", default='1RslmRUjzYLxgpPyzSs1BP3ggCXGJRgdj', help="Download link")
    parser.add_argument("--eval_period", default=50, help="Evaluation iteration number", type = int)
    parser.add_argument("--resume_training", default=False, help="Start training from last weights")
    parser.add_argument("--project", default='number_plate', help="Give project name")

    opt = parser.parse_args()
    # print(opt)

    run_command = 'python train.py  --data_source '+str(opt.data_source)+' --batch '+str(opt.batch)+  ' --iterations '+str(opt.iterations) +' --num_gpus ' +str(opt.num_gpus)+' --cfg_model ' +str(opt.cfg_model) + ' --eval_period ' +str(opt.eval_period) + ' --resume_training '+str(opt.resume_training) +' --project '+str(opt.project)
    print(run_command)



    p = subprocess.Popen(run_command, shell=True, stdout=subprocess.PIPE)
    # latest_count = 1
    while True:
        out = p.stdout.readline()

        if out.decode("utf-8").find("total_loss")!=-1:
            print((out.decode("utf-8")))
            terminal = out.decode("utf-8")


            itera_no = terminal.split('iter: ')
            print('Current Iteration :',itera_no[1].split(' ')[0])
            start_plot()
            # Add a second sleep Here
            time.sleep(1)

            send_tele(out.decode("utf-8"),'Plot.png')

            start = terminal.find('Starting training')
            if start != -1:
                updater = Updater('805281461:AAH09xnakEe8MxOBLQ7jWaiNolGxoZyxxrM')
                dp = updater.dispatcher
                token='805281461:AAH09xnakEe8MxOBLQ7jWaiNolGxoZyxxrM'
                bot = telegram.Bot(token=token)
                bot.send_message(chat_id='-496362146', text = '   TRAINING HAS BEEN STARTED  ')
            end = terminal.find('Total training time:')
            if end != -1:
                updater = Updater('805281461:AAH09xnakEe8MxOBLQ7jWaiNolGxoZyxxrM')
                dp = updater.dispatcher
                token='805281461:AAH09xnakEe8MxOBLQ7jWaiNolGxoZyxxrM'
                bot = telegram.Bot(token=token)
                bot.send_message(chat_id='-496362146', text = '   TRAINING HAS BEEN ENDED  ')













