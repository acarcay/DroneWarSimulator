import heapq, random, time

class LossyChannel:
    def __init__(self, base_delay=0.03, jitter=0.01, loss_p=0.05):
        self.q = []
        self.base_delay = base_delay
        self.jitter = jitter
        self.loss_p = loss_p

    def send(self, msg):
        # belli ihtimalle kayıp
        if random.random() < self.loss_p:
            return
        delay = max(0.0, random.gauss(self.base_delay, self.jitter))
        heapq.heappush(self.q, (time.time() + delay, msg))

    def recv_ready(self):
        now = time.time()
        out = []
        while self.q and self.q[0][0] <= now:
            out.append(heapq.heappop(self.q)[1])
        return out