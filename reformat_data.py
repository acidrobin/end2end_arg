import pandas as pd
import os 
import os.path as op


def convert_comments_to_string(thread):
    topic = thread.iloc[0]["title"]

    comments_list = list(thread["comment"])

    comment_numbers = [f"Comment {x}: " for x in range(1,len(comments_list)+1)]

    numbered_comments = [x + y for x,y in list(zip(comment_numbers, comments_list))]

    comments_as_string = ("\n\n".join(numbered_comments))

    comments_as_string = f"Main topic: {topic}\n\n" + comments_as_string

    return comments_as_string

def convert_summaries_to_string(thread):
    topic = thread.iloc[0]["title"]

    comments_list = list(thread["comment"])

    comment_numbers = [f"Comment {x}" for x in range(1,len(comments_list)+1)]

    comment_stance = list(thread["stance"])

    comment_summaries = list(thread["summary"])

    numbered_summaries = [f"{number} ({stance}): {summary}" for number, stance, summary in list(zip(comment_numbers, comment_stance, comment_summaries))]
    
    summaries_as_string = ("\n\n".join(numbered_summaries))

    return summaries_as_string

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

        thread_csv = pd.read_csv(op.join("debatabase_data","threads",name))
        #Comments are shuffled so that the llm cannot learn the simple pattern in the
        #comments (alternating attack and support)
        csv_shuffled = thread_csv.sample(frac=1, random_state=42)
        new = csv_shuffled[csv_shuffled["comment_id"].str.startswith("p")]
        

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
    
    summaries_df_train.to_csv("debatabase_data/end_to_end_train.csv")
    summaries_df_val.to_csv("debatabase_data/end_to_end_val.csv")
    summaries_df_test.to_csv("debatabase_data/end_to_end_test.csv")


generate_end_to_end()