import pandas as pd
import os 
import os.path as op

import numpy as np
np.random.seed(42)

from nltk import sent_tokenize

def first_sent_summarize(comment):
    return sent_tokenize(comment)[0]


def convert_comments_to_string(thread):
    topic = thread.iloc[0]["title"]

    comments_list = list(thread["comment"])

    comment_numbers = [f"Comment {x}: " for x in range(1,len(comments_list)+1)]

    numbered_comments = [x + y for x,y in list(zip(comment_numbers, comments_list))]

    comments_as_string = ("\n\n".join(numbered_comments))

    comments_as_string = f"Main topic: {topic}\n\n" + comments_as_string

    return comments_as_string


def get_parent_label(thread, parent_id):
    if parent_id.startswith("t"):
        return "main topic"
    else:
        label_num = thread.index[thread["comment_id"] == parent_id].item() +1
        return f"Comment {label_num}"

def convert_summaries_to_string(thread):
    topic = thread.iloc[0]["title"]

    comments_list = list(thread["comment"])

    comment_numbers = [f"Comment {x}" for x in range(1,len(comments_list)+1)]

    comment_stance = list(thread["stance"])

    comment_summaries = list(thread["summary"])

    parent_labels = [get_parent_label(thread, p) for p in list(thread["parent_id"])]

    numbered_summaries = [f"{number} ({stance}s {p_lab}): {summary}" for number, stance, p_lab, summary in 
                                    list(zip(comment_numbers, comment_stance, parent_labels, comment_summaries))]
    
    summaries_as_string = ("\n\n".join(numbered_summaries))

    return summaries_as_string



def is_valid(sequence, parents_list):
    """Parents_list: [(comment_id, parent_id)]  """
    return all([sequence.index(comment_id) > sequence.index(parent_id) 
                for comment_id, parent_id in parents_list])


def parents_valid(sequence, parents_list):
    """Parents_list: [(comment_id, parent_id)]  """
    return all([p in sequence for c, p in parents_list if c in sequence])


def generate_end_to_end():

    thread_names = [name for name in os.listdir(op.join("debatabase_data","threads")) if name.endswith(".csv")]
    topics_df = pd.read_csv("debatabase_data/topics_df.csv")

    # topics_df["topic"] = topics_df["topic"].apply(lambda x: x.strip("."))

    test_names = topics_df[topics_df["set"] == "test"]["topic"].to_list()
    train_names = topics_df[topics_df["set"] == "train"]["topic"].to_list()
    val_names = topics_df[topics_df["set"] == "val"]["topic"].to_list()



    cs_train = []
    ss_train = []

    cs_val = []
    ss_val = []

    cs_test = []
    ss_test = []

    for name in thread_names:

        np.random.seed(np.random.choice(100,1))

        thread_csv = pd.read_csv(op.join("debatabase_data","threads",name))


        thread_csv["first_sent"] = thread_csv.comment.map(first_sent_summarize)
        thread_csv.loc[thread_csv.summary == "no summary", "summary"] = thread_csv.first_sent


        #Comments are shuffled so that the llm cannot learn the simple pattern in the
        #comments (alternating attack and support)
        parents_csv = thread_csv[thread_csv["parent_id"].str.startswith("p")]

        parents_list = list(zip(list(parents_csv.comment_id), list(parents_csv.parent_id)))
        ids_list = list(thread_csv.comment_id)

        np.random.shuffle(ids_list)
        while not is_valid(ids_list, parents_list):
            # csv_shuffled = csv_shuffled.sample(frac=1, random_state=random_state)
            np.random.shuffle(ids_list)


        #randomly delete all but six comments
        short_list = np.delete(ids_list,np.random.choice(range(len(ids_list)),max(len(ids_list)-6,0),replace=False))
        while not parents_valid(short_list, parents_list):
            short_list = np.delete(ids_list,np.random.choice(range(len(ids_list)),max(len(ids_list)-6,0),replace=False))

        new = thread_csv.set_index("comment_id").loc[short_list].reset_index()
        

        comments_as_string = convert_comments_to_string(new)
        summaries_as_string = convert_summaries_to_string(new)

        name = name[:-4]
        print(name)

        if name in test_names:
            cs_test.append(comments_as_string)
            ss_test.append(summaries_as_string)
        elif name in val_names:
            cs_val.append(comments_as_string)
            ss_val.append(summaries_as_string)
        elif name in train_names:
            cs_train.append(comments_as_string)
            ss_train.append(summaries_as_string)
        else:
            raise ValueError("The name should be either in train, test or val")
            

    summaries_df_train = pd.DataFrame({"comments": cs_train,
                                    "summaries": ss_train})
    
    summaries_df_val = pd.DataFrame({"comments": cs_val,
                                    "summaries": ss_val})
    
    summaries_df_test = pd.DataFrame({"comments": cs_test,
                                    "summaries": ss_test})
    
    summaries_df_train.to_csv("debatabase_data/end_to_end_train_multilevel.csv")
    summaries_df_val.to_csv("debatabase_data/end_to_end_val_multilevel.csv")
    summaries_df_test.to_csv("debatabase_data/end_to_end_test_multilevel.csv")


generate_end_to_end()