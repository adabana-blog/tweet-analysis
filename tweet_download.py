import requests
import pandas as pd
import json
import ast
import yaml


def get_tweet_id():
    path = "../../../data/twitter_analysis/tweets_open.csv.bz2"
    df_tweets = pd.read_csv(path, header=None)
    df_tweets = df_tweets.rename(columns={0: 'index' ,1: 'category', 2: 'id', 3: 'posi_and_nega', 4: 'posi', 5: 'nega', 6: 'neutral', 7: 'Irrelevant'})
    return df_tweets


def create_twitter_url(id_list):
    #tweet_fields = "tweet.fields=lang,author_id"
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    str_a = "ids="
    str_id_list = [str(x) for x in id_list]
    str_b = ",".join(str_id_list)
    ids = str_a + str_b
    # You can adjust ids to include a single Tweets.
    # Or you can add to up to 100 comma-separated IDs
    url = "https://api.twitter.com/2/tweets?{}".format(ids)
    return url


def process_yaml():
    with open("config.yaml") as file:
        return yaml.safe_load(file)


def create_bearer_token(data):
    return data["search_tweets_api"]["bearer_token"]


def twitter_auth_and_connect(bearer_token, url):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    response = requests.request("Get", url, headers=headers)
    return response.json()


def lang_data_shape(res_json):
    data_only = res_json["data"]
    doc_start = '"documents": {}'.format(data_only)
    str_json = "{" + doc_start + "}"
    dump_doc = json.dumps(str_json)
    doc = json.loads(dump_doc)
    return ast.literal_eval(doc)


def save_jsonfile(file, i):
    path = 'text/text_' + str(i) + '.json'
    fw = open(path,'w')
    json.dump(file,fw,indent=4)


def create_dataframe():
    df_text = None
    for i in range(1, 239800, 100):
        path = "text/text_" + str(i)+ ".json"
        with open(path) as f:
            jsonfile = json.load(f)
            tweet_id_list = []
            tweet_text_list = []
            for row in jsonfile["documents"]:
                tweet_id_list.append(int(row["id"]))
                tweet_text_list.append(row["text"])
            if i == 1:
                df_text = pd.DataFrame({"id": tweet_id_list, "text": tweet_text_list})
            else:
                df_text = pd.concat([df_text, pd.DataFrame({"id": tweet_id_list, "text": tweet_text_list})], axis=0)
    return df_text


def main():
    data = process_yaml()
    create_bearer_token(data)
    bearer_token = create_bearer_token(data)
    df_tweets = get_tweet_id()
    status_id_list = list(df_tweets[2])
    #全データ数53496300のうち23980000件出力済み
    for i in range(239801, len(status_id_list), 100):
        tmp_id_list = status_id_list[i:i+100]
        url = create_twitter_url(tmp_id_list)
        res_json = twitter_auth_and_connect(bearer_token, url)
        jsonfile = lang_data_shape(res_json)
        save_jsonfile(jsonfile, i)
        print(i)
    df_text = create_dataframe()
    df_tweets = df_tweets.merge(df_text, on="id", how="inner")
    df_tweets.to_csv("data/df_tweets")