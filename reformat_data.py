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

    cs = []
    ss = []

    for name in thread_names:

        thread_csv = pd.read_csv(op.join("debatabase_data","threads",name))
        csv_shuffled = thread_csv.sample(frac=1, random_state=42)
        new = csv_shuffled[csv_shuffled["comment_id"].str.startswith("p")]
        comments_as_string = convert_comments_to_string(new)
        summaries_as_string = convert_summaries_to_string(new)

        cs.append(comments_as_string)
        ss.append(summaries_as_string)


    summaries_df = pd.DataFrame({"comments": cs,
                                 "summaries": ss})
    
    summaries_df.to_csv("debatabase_data/end_to_end.csv")

generate_end_to_end()