import pandas as pd
import os 
import os.path as op

thread_names = [name for name in os.listdir(op.join("debatabase_data","threads")) if name.endswith(".csv")]

for name in thread_names:


    thread_csv = pd.read_csv(op.join("debatabase_data","threads",name))

    csv_shuffled = thread_csv.sample(frac=1)

    new = csv_shuffled[csv_shuffled["comment_id"].str.startswith("p")]


    print(new)
    comments_as_string = ("\n\n".join(list(new["comment"])))

    print(comments_as_string)
