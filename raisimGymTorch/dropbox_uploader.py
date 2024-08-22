import argparse

import dropbox
import os


class TransferData:
    def __init__(self, access_token):
        self.access_token = access_token

    def upload_files(self, input_dir, output_dir):
        """upload a file to Dropbox using API v2
        """
        dbx = dropbox.Dropbox(self.access_token)

        for dir, dirs, files in os.walk(input_dir):
            for file in files:
                try:
                    file_path = os.path.join(dir, file)
                    dest_path = os.path.join(output_dir+dir, file)
                    print(f"Uploading {file_path} to {dest_path}")
                    with open(file_path, "rb") as f:
                        dbx.files_upload(f.read(), dest_path, mute=True)
                except Exception as err:
                    print(f"Failed to upload {file}\n{err}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='')

    parser.add_argument('--input_dir',
                        help='Directory to copy',
                        default=None,
                        type=str)
    parser.add_argument('--dropbox_dir',
                        help='Dropbox directory',
                        default=None,
                        type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    access_token = "ad1Di4LSFBgAAAAAAAAAAWetJg4CBKF7NbNzPwZ3HcoJsxCFK8tLYG1JtPYGrrGV"
    transfer_data = TransferData(access_token)

    input_dir = args.input_dir
    output_dir = args.dropbox_dir

    transfer_data.upload_files(input_dir, output_dir)


if __name__ == '__main__':
    main()
