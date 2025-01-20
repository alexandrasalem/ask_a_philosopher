import json
import re

def split_to_dict_list(text, text_name, chapter_re, book_re=None, end_re=None):
    first_dict_list = []
    first_dict_template = {
        "text_name": text_name,
        "book_label": "",
        "chapter_label": "",
        "chapter_text": ""
    }
    chapter = []
    chapter_label = None
    book_label = None
    update_book_label = None
    for line in text:
        # if we've provided a regex for book, then look for that
        if book_re:
            if re.match(book_re, line):
                update_book_label = re.match(book_re, line)[0][:-1]
        # we've reached an extra section at the end
        if end_re:
            if re.match(end_re, line):
                if not chapter_label:
                    continue
                else:
                    break
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
                if not book_label:
                    book_label = update_book_label
                new_dict["book_label"] = book_label
                new_dict["chapter_label"] = chapter_label
                new_dict["chapter_text"] = "".join(chapter)
                first_dict_list.append(new_dict)
                # and then start over
                chapter_label = re.match(chapter_re, line)[0][:-1]
                book_label = update_book_label
                chapter = []
        else:
            chapter.append(line)

    # now final one
    first_dict_template["book_label"] = book_label
    first_dict_template["chapter_label"] = chapter_label
    first_dict_template["chapter_text"] = "".join(chapter)
    first_dict_list.append(first_dict_template)
    return first_dict_list

all_text_dict_list = []

with open("data_aristotle/1974.txt", "r") as f:
    first_text = f.readlines()

chapter_re = "^[IXV]+\n"
test = split_to_dict_list(first_text, "Poetics", chapter_re)
all_text_dict_list.extend(test)

with open("data_aristotle/2412.txt", "r") as f:
    first_text = f.readlines()

book_re = "^Section \d+\n"
chapter_re = "^Part \d+\n"
test = split_to_dict_list(first_text,"Categories", chapter_re, book_re)
all_text_dict_list.extend(test)

with open("data_aristotle/6762.txt", "r") as f:
    first_text = f.readlines()

book_re = "^BOOK [IXV]+\n"
chapter_re = "^CHAPTER [IXV]+\n"
end_re = "^INDEX\n"
test = split_to_dict_list(first_text,"Politics", chapter_re, book_re, end_re)
all_text_dict_list.extend(test)

with open("data_aristotle/8438.txt", "r") as f:
    first_text = f.readlines()

book_re = "^BOOK [IXV]+\n"
chapter_re = "^Chapter [IXV]+.\n"
end_re = "^NOTES\n"
test = split_to_dict_list(first_text,"Nicomachean Ethics", chapter_re, book_re, end_re)
all_text_dict_list.extend(test)

with open("data_aristotle/26095.txt", "r") as f:
    first_text = f.readlines()

chapter_re = "^Part \d+\n"
end_re = "^THE END$"
test = split_to_dict_list(first_text,"The Athenian Constitution", chapter_re=chapter_re, end_re=end_re)
all_text_dict_list.extend(test)

with open("data_aristotle/59058.txt", "r") as f:
    first_text = f.readlines()

book_re = "^BOOK THE [A-Z]+.\n"
chapter_re = "^CHAPTER [IXV]+.\n"
end_re = "^INDEX.$"
test = split_to_dict_list(first_text,"History of Animals", chapter_re=chapter_re, book_re=book_re, end_re=end_re)
all_text_dict_list.extend(test)


with open("data_aristotle/67858.txt", "r") as f:
    first_text = f.readlines()

book_re = "(^\s+THE MASTER\-PIECE\.\n)|(^\s+THE MIDWIFE\.\n)"
chapter_re = "(^\s+CHAPTER [IXV]+.\n)|(^\s+BOOK I.—CHAPTER I.)"
end_re = "^\s+THE END.\n"
test = split_to_dict_list(first_text[180:],"Aristotle's Works", book_re=book_re, chapter_re=chapter_re, end_re=end_re)
all_text_dict_list.extend(test)


# Serializing json
json_object = json.dumps(all_text_dict_list, indent=4)
#
# # Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)