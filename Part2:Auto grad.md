```python
import  torch
'''
1.自动求导
'''
#设置自动求导
x=torch.ones(2,2,requires_grad=True)
#对张量进行运算
y=x+2
out=(y*y+3).mean()
'''
2.梯度
'''
#对out反向传播
out.backward()
#输出d(out)/dx
print(x.grad)
#创建结果为向量的计算过程
x=torch.randn(3,requires_grad=True)
y=x*2
while y.data.norm()<1000:
    y=y*2
#计算d(y)/dv,v=[0.1,1.0,0.0001]
v=torch.tensor([0.1,1.0,0.0001],dtype=torch.float)
y.backward(v)
print(x.grad)
#关闭梯度记录
with torch.no_grad():
    print((x**2).requires_grad)
```
