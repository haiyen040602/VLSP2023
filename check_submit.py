import sys
import json
import zipfile

TEST_FILES = [f'test_{i:04d}.txt' for i in range(1, 37)]
SENTENCE_NUMBERS = [90, 73, 113, 144, 107, 54, 69, 148, 59, 96, 60, 17, 239, 88, 51, 69, 202, 62, 65, 48, 71, 73, 124, 81, 40, 235, 61, 129, 126, 28, 54, 84, 54, 123, 63, 69]

def read_file(f):
    data = []
    sent_tuples = []
    txt = False
    for l in f:
        l = l.decode('utf-8').strip()

        if len(l) == 0:
            if txt:
                data.append(sent_tuples)
            sent_tuples = []
            txt = False
        elif l.startswith('{'):
            sent_tuples.append(json.loads(l))
        else:
            # text line
            txt = True

    if txt:
        data.append(sent_tuples)

    return data

def check(zip_file_name):
    # Check if all files exist
    zip_file = zipfile.ZipFile(zip_file_name, 'r')
    zip_file_list = zip_file.namelist()
    missing_files = [file for file in TEST_FILES if file not in zip_file_list]

    if not missing_files:
        print('All test files exist in the ZIP archive.')
        
        # Check and read each file
        passed = True
        for file_name, sent_num in zip(TEST_FILES, SENTENCE_NUMBERS):
            try:
                with zip_file.open(file_name, 'r') as file:
                    data = read_file(file)

                    if len(data) != sent_num:
                        raise ValueError('The number of sentences does not match.')
            except Exception as e:
                passed = False
                print(f'Error in file {file_name}: {e}')
        
        if passed:
            print('PASSED')
    else:
        print('Missing files:', missing_files)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        zip_file_name = sys.argv[1]
        check(zip_file_name)
    else:
        print('No command-line arguments provided. Input a submit zip file to check.')