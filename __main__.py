"""
Connects to Botzone
"""
import urllib.request
import time
import json


class Config:
    url = "https://www.botzone.org.cn/api/65605cc282ee46246bb02d05/8996560798/localai"
    # url = "https://www.botzone.org.cn/api/65605cc282ee46246bb02d05/8996560798/runmatch"


URL = {
    "localai": "https://www.botzone.org.cn/api/65605cc282ee46246bb02d05/8996560798/localai",
    "runmatch": "https://www.botzone.org.cn/api/65605cc282ee46246bb02d05/8996560798/runmatch" 
}


class Match:
    has_request = False
    has_response = False
    current_request = None
    current_response = None
    matchid = None

    def new_request(self, request):
        self.has_request = True
        self.has_response = False
        self.current_request = request


# TODO：定义一种特化的对局数据类，比如存储棋盘状态等


class SomeKindOfMatch(Match):
    def __init__(self, matchid, first_request):
        self.has_request = True
        self.current_request = first_request
        self.matchid = matchid


headers = {
    "X-Game": "Gomoku",
    "X-Player-0": "me",
    "X-Player-1": "654a0176ec1ab1389703ab4b",
}

matches = {}


# 从 Botzone 上拉取新的对局请求
def fetch(matchClass, mode="localai"):
    req = urllib.request.Request(URL[mode], headers=headers)
    for matchid, m in matches.items():
        if m.has_response and m.has_request and m.current_response:
            print('> Response for match [%s]: %s' % (matchid, m.current_response))
            m.has_request = False
            req.add_header("X-Match-" + matchid, m.current_response)
    while True:
        try:
            res = urllib.request.urlopen(req, timeout=None)
            botzone_input = res.read().decode()
            lines = botzone_input.split('\n')
            if mode == "localai":
                request_count, result_count = map(int, lines[0].split(' ')) # '658a3a1026354d2b0ff4f858'
                for i in range(0, request_count):
                    # 新的 Request
                    matchid = lines[i * 2 + 1]
                    request = lines[i * 2 + 2]
                    if matchid in matches:
                        print('Request for match [%s]: %s' % (matchid, request))
                        matches[matchid].new_request(request)
                    else:
                        print('New match [%s] with first request: %s' % (matchid, request))
                        matches[matchid] = matchClass(matchid, request)
                for i in range(0, result_count):
                    # 结束的对局结果
                    matchid, slot, player_count, *scores = lines[request_count * 2 + 1 + i].split(' ')
                    if player_count == "0":
                        print("Match [%s] aborted:\n> I'm player %s" % (matchid, slot))
                    else:
                        print("Match [%s] finished:\n> I'm player %s, and the scores are %s" % (matchid, slot, scores))
                    matches.pop(matchid)
            else:
                raise NotImplementedError
                    
        except (urllib.error.URLError, urllib.error.HTTPError):
            # 此时可能是长时间没有新的 request 导致连接超时，再试即可
            print("Error reading from Botzone or timeout, retrying 5 seconds later...")
            time.sleep(5)
            continue
        break


if __name__ == '__main__':
    import json
    import time
    import numpy as np
    from rl_gomoku.envs import ArrayGomoku
    from rl_gomoku.agents import GreedyAgent, MCTSZeroAgent
    from rl_gomoku.utils import create_model_from_args, default_args
    model_args, train_args = default_args()
    cache_data = json.load(open("./cache_data/quick_list_set.json", "r"))
    model_args.update({
        "device": "cuda:0",
        "model_type": "res5",
    })
    model = create_model_from_args(model_args)
    # model.load_checkpoint("/home/nymath/dev/rl/gomoku/model/ResNet3-gomuku-2023-12-24-17-26-iteration-9000.pth")
    model.load_checkpoint("/home/nymath/dev/rl/gomoku/model/ResNet5-gomuku-2023-12-24-14-27-iteration-44000.pth")
    board_size = 15
    env = ArrayGomoku(board_size=board_size)
    action = np.array([-1, -1])
    agent = MCTSZeroAgent(model.forward, player_id=1, c_puct=5, n_playout=2000)
    # agent = GreedyAgent(board_size, player_id=1, cache_data=cache_data, random_state=12345)

    first_round = True
    x, y = -1, -1

    env.reset()
    agent.reset()
    
    while True:
        fetch(SomeKindOfMatch, mode="localai")
        # req = urllib.request.Request(Config.url, headers=headers)
        # res = urllib.request.urlopen(req, timeout=None)
        # botzone_input = res.read().decode()
        # lines = botzone_input.split('\n')
        # match_id = lines[0]
        # m = SomeKindOfMatch(match_id, json.dumps({"x":int(-1), "y":int(-1)}))
        # m.current_response = json.dumps({"x": int(7), "y": int(7)})
        # req = urllib.request.Request(Config.url, headers=headers)
        # req.add_header("X-Match-" + match_id, m.current_response)
        
        # m.new_request(json.dumps({"x":7, "y":7}))
        
        for mid, m in matches.items():
            # 使用 m.current_request 模拟一步对局状态，然后产生动作
            # m.current_request = {"response": {"x": 7, "y": 7}}
            # if first_round:
            requests = json.loads(m.current_request)
            # responses = json.loads(m.current_response)
            x, y = requests["x"], requests["y"]
            if x != -1:
                first_round = False
                position_idx = y * board_size + x
                env.step(position_idx)

            position_idx = agent.get_action(env)
            (y, x) = position_idx // board_size, position_idx % board_size
            env.step(position_idx)
            m.current_response = json.dumps({"x": int(x), "y": int(y)})
            print(json.dumps({"response": {"x": int(x), "y": int(y)}}))
            # 将自己的动作存入 m.current_response，同样进行一步模拟
            m.has_response = True
            env.render()
