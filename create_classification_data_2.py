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

def get_parent_comment(thread, parent_id):
    if parent_id.startswith("t"):
        return thread.iloc[0]["title"]
    else:
        comment = thread.iloc[thread.index[thread["comment_id"] == parent_id].item()]["comment"]
        return comment


def get_classification_triples(thread):
    topic = thread.iloc[0]["title"]

    comments_list = list(thread["comment"])

    parents_list = [topic] + comments_list

    # parents_list = [get_parent_comment(thread, p) for p in list(thread["parent_id"])]

    comment_numbers = [f"Comment {x}" for x in range(1,len(comments_list)+1)]

    comment_stance = list(thread["stance"])

    comment_summaries = list(thread["summary"])

    parent_labels = [get_parent_label(thread, p) for p in list(thread["parent_id"])]

    parent_numbers = ["main topic"] + comment_numbers

    parent_comments = [get_parent_comment(thread, p) for p in list(thread["parent_id"])]

    children_out = []
    parents_out = []
    stance_labels = []


    for comment_num, comment, p_lab, stance in list(zip(comment_numbers, comments_list, parent_labels, comment_stance)) :
        for parent_num, parent in zip(parent_numbers, parents_list):
            if comment_num != parent_num:
                children_out.append(comment)
                parents_out.append(parent)

                if p_lab.strip() == parent_num.strip():
                    stance_labels.append(stance)
                else:
                    stance_labels.append("none")

    return {"child":children_out,"parent":parents_out, "stance":stance_labels}



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


    train_triples = {"child":[],"parent":[],"stance":[]}

    val_triples = {"child":[],"parent":[],"stance":[]}

    test_triples = {"child":[],"parent":[],"stance":[]}




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
        

        triples = get_classification_triples(new)

        name = name[:-4]
        print(name)

        if name in test_names:
            test_triples["child"] += triples["child"]
            test_triples["parent"] += triples["parent"]
            test_triples["stance"] += triples["stance"]

        elif name in val_names:
            val_triples["child"] += triples["child"]
            val_triples["parent"] += triples["parent"]
            val_triples["stance"] += triples["stance"]

        elif name in train_names:
            train_triples["child"] += triples["child"]
            train_triples["parent"] += triples["parent"]
            train_triples["stance"] += triples["stance"]

        else:
            raise ValueError("The name should be either in train, test or val")
            

    df_train = pd.DataFrame(train_triples)
    
    df_val = pd.DataFrame(val_triples)
    
    df_test = pd.DataFrame(test_triples)


    df_train.to_csv("debatabase_data/classification_train_multilevel.csv")
    df_val.to_csv("debatabase_data/classification_val_multilevel.csv")
    df_test.to_csv("debatabase_data/classification_test_multilevel.csv")


    # summaries_df_train.to_csv("debatabase_data/end_to_end_train_multilevel.csv")
    # summaries_df_val.to_csv("debatabase_data/end_to_end_val_multilevel.csv")
    # summaries_df_test.to_csv("debatabase_data/end_to_end_test_multilevel.csv")


generate_end_to_end()