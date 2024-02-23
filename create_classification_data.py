import pandas as pd
import os 
import os.path as op


def get_classification_triples(thread):


    topic = thread.iloc[0]["title"]

    comments_list = list(thread["comment"])

    # comment_numbers = [f"Comment {x}" for x in range(1,len(comments_list)+1)]

    comment_stance = list(thread["stance"])

    return {"child": comments_list, "parent": [topic]*len(comment_stance),"stance":comment_stance}


def generate_end_to_end():

    thread_names = [name for name in os.listdir(op.join("debatabase_data","threads")) if name.endswith(".csv")]
    topics_df = pd.read_csv("debatabase_data/topics_df.csv")

    # topics_df["topic"] = topics_df["topic"].apply(lambda x: x.strip("."))

    test_names = topics_df[topics_df["set"] == "test"]["topic"].to_list()
    train_names = topics_df[topics_df["set"] == "train"]["topic"].to_list()
    val_names = topics_df[topics_df["set"] == "val"]["topic"].to_list()

    comments_word_count = 0
    summaries_word_count = 0
    n_comments = 0
    train_triples = {"child":[],"parent":[],"stance":[]}

    val_triples = {"child":[],"parent":[],"stance":[]}

    test_triples = {"child":[],"parent":[],"stance":[]}

    for name in thread_names:

        thread_csv = pd.read_csv(op.join("debatabase_data","threads",name))
        
        #Comments are shuffled so that the llm cannot learn the simple pattern in the
        #comments (alternating attack and support)
        csv_shuffled = thread_csv.sample(frac=1, random_state=42)
        new = csv_shuffled[csv_shuffled["comment_id"].str.startswith("p")]

        triples = get_classification_triples(new)
        print(triples)

        name = name[:-4]

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


    df_train.to_csv("debatabase_data/classification_train.csv")
    df_val.to_csv("debatabase_data/classification_val.csv")
    df_test.to_csv("debatabase_data/classification_test.csv")


generate_end_to_end()