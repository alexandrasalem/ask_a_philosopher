import json
import re

def split_to_dict_list_no_books(text_name, chapter_re, book_re=None):
    first_dict_list = []
    first_dict_template = {
        "text_name": text_name,
        "book_label": "1",
        "chapter_label": "",
        "chapter_text": ""
    }
    chapter = []
    chapter_label = None
    book_label = None
    for line in first_text:
        # if we've provided a regex for book, then look for that
        if book_re:
            if re.match(book_re, line):
                book_label = re.match(book_re, line)[0][:-1]
        # this if statement should only occur at the very beginning
        if not chapter_label and not re.match(chapter_re, line):
            continue
        elif re.match(chapter_re, line):
            # this indicates this is the first chapter
            if not chapter_label:
                chapter_label = re.match(chapter_re, line)[0][:-1]
                continue
            else:
                new_dict = first_dict_template.copy()
                # here we add that chapter to the dictionary
                new_dict["book_label"] = book_label
                new_dict["chapter_label"] = chapter_label
                new_dict["chapter_text"] = "".join(chapter)
                first_dict_list.append(new_dict)
                # and then start over
                chapter_label = re.match(chapter_re, line)[0][:-1]
                chapter = []
        else:
            chapter.append(line)

    # now final one
    first_dict_template["book_label"] = book_label
    first_dict_template["chapter_label"] = chapter_label
    first_dict_template["chapter_text"] = "".join(chapter)
    first_dict_list.append(first_dict_template)
    return first_dict_list

# with open("data_aristotle/1974.txt", "r") as f:
#     first_text = f.readlines()

# chapter_re = "^[IXV]+\n"
# test = split_to_dict_list_no_books("Poetics", chapter_re)

with open("data_aristotle/2412.txt", "r") as f:
    first_text = f.readlines()

book_re = "^Section \d+\n"
chapter_re = "^Part \d+\n"
test = split_to_dict_list_no_books("Categories", chapter_re, book_re)

# Serializing json
json_object = json.dumps(test, indent=4)
#
# # Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)