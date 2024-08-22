import os
import time
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import html
import subprocess
import re

slack_app_token = "xapp-1-A02T0RTG7NG-2916654241811-cd7a651093ad18fc27622b84d1b543843b3794ad5b7713e8f5b64b3975abecd5"
slack_bot_token = "xoxb-2929055975057-2929242430049-83r1d9ugvK6Y9iY7C5wIe6BU"

# Initialize app with bot token and socket mode handler
app = App(token=slack_bot_token)

@app.message(":command:")
def command_listener(message, say):
    # Works with text :command: bash command
    user = message['user']
    command = html.unescape(message['text'].replace(":command: ", ""))

    if message.get("thread_ts"):
        thread = message["thread_ts"]
    else:
        thread = message["ts"]

    # Just to avoid messing up in the cluster
    if re.search("(^| )rm($| )", command):
        say({"text": f"Error invalid command!", "thread_ts": thread})
        return

    try:
        output = subprocess.check_output(
            command, stderr=subprocess.STDOUT, shell=True).decode('UTF-8')
    except subprocess.CalledProcessError as exc:
        say({"text": f"FAIL: {exc.returncode} \n```\n{exc.output}```", "thread_ts": thread})
    else:
        say({"text": f"Output: \n```\n{output}```", "thread_ts": thread})


if __name__ == "__main__":
    SocketModeHandler(app, slack_app_token).start()
