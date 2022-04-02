import os
import lib_api
import json


# def write_answer(file_name, folder_to_save_answers, answer):
#     file_name_without_ext, _ = os.path.splitext(file_name)
#     ext = '.txt'
#     full_file_to_write_path = os.path.join(folder_to_save_answers, 'answer_' + file_name_without_ext + ext)
#     with open(full_file_to_write_path, 'w') as f:
#         f.write('1' if answer == True else '0')


def count_accuracy(folder_with_images, should_objects_fit, all_answers):
    files = os.listdir(folder_with_images)
    num_of_correct_answers = 0
    for file in files:
        full_file_path = os.path.join(folder_with_images, file)

        can_fit = lib_api.check_image(full_file_path)

        all_answers[file] = can_fit

        if should_objects_fit == can_fit:
            num_of_correct_answers += 1

    return num_of_correct_answers, len(files)


def process():
    folder_with_fit_images = 'D:\\signal processing\\big_lab\\Intelligent-Placer\\pictures\\input_fit'
    folder_with_not_fit_images = 'D:\\signal processing\\big_lab\\Intelligent-Placer\\pictures\\input_not_fit'
    folder_to_save_answers = 'D:\\signal processing\\big_lab\\Intelligent-Placer\\answers'

    all_answers = {}
    all_answers_file_name = 'all_answers.json'

    num_of_fit_correctly, num_of_should_fit_correctly = count_accuracy(folder_with_fit_images, True, all_answers)
    num_of_not_fit_correctly, num_of_should_not_fit_correctly = count_accuracy(folder_with_not_fit_images, False,
                                                                               all_answers)

    accuracy = (num_of_fit_correctly + num_of_not_fit_correctly) / (
                num_of_should_fit_correctly + num_of_should_not_fit_correctly)

    print('Accuracy: ' + str(accuracy))

    with open(os.path.join(folder_to_save_answers, all_answers_file_name), 'w') as f:
        json.dump(all_answers, f)


if __name__ == '__main__':
    process()
