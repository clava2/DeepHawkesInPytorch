from clickhouse_driver import Client

if __name__ == "__main__":
    client = Client(host='121.43.42.31',port='4118',user="default" ,password="123",database = "weibo")
    sql = 'select * from weibo where reposts_count > 100;'
    ans = client.execute(sql)
    ans = set([a[-1] if a[-1] != '' else a[0] for a in ans])
    for a in ans:
        graph_data_sql = "select id,user_id,created_at,retweet_id from weibo where id == '" + a + "' or retweet_id == '" + a + "';"
        ans = client.execute(graph_data_sql)
        print(ans)