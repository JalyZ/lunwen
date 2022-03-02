import torch
import numpy as np

M = np.zeros([20,20],dtype=int)
M[4,5] = 1
M[14,10] = 1
M[7,15] = 1
safeDistance = 3
punishment = -100
end = np.array([[4,5],[14,10],[7,15]])

initState = np.array([[0,0],[19,0],[0,19]])

def distanceCalculate(state):
    num = state.shape[0]
    d = []
    for i in range(num):
        for j in range(i+1,num):
            dis = np.linalg.norm(state[i]-state[j])
            d.append(dis)
    return d

r = 0
for s in initState:
    if min(s[0],s[1])<0 or max(s[0],s[1])>19:
        s[0] = np.clip(s[0],0,19)
        s[1] = np.clip(s[1],0,19)
        r += -100
    if min(distanceCalculate(end))<safeDistance:
        r += punishment
    r += M[s[0],s[1]]# - np.linalg.norm(np.array([14,10])-s)
print(r)


loss = torch.nn.MSELoss(reduction='sum')
pred=torch.tensor([1.1,2.2],dtype=torch.float)
target = torch.tensor([1,2],dtype=torch.float)
output = loss(pred, target)
print(output)
