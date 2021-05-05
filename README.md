

```python
import numpy as np
import pandas as pd
import math
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt
plt.style.use('ggplot')
font = {'family' : 'meiryo'}
```

# 漸近展開の備忘録


## 1. 漸近級数展開

漸近級数を用いて関数を近似する際にはおよそ以下のステップを踏んで近似式を求める。

- ステップ①：関数を漸近級数の形で表す

- ステップ②：絶対値が最小となる第k項を特定する

- ステップ③：第k+1項以降を残差として、第k項までの有限級数で打ち切る


### 1-1. 漸近展開の例

ここでは例として、以下の関数$F_0(x)$の漸近級数を求めて、近似式を特定する。

$$
F_0(x) = e^{1/x} \int^{\infty}_{1/x}{e^{-t}t^{-1}}
$$


```python
def F_0(x):
    return np.exp(1.0/x) * integrate.quad(lambda t: np.exp(-t) * t**(-1), 1.0/x, 1000.0 )[0]
```


```python
F_0(0.1)
```




    0.09156333394018187



この関数は、漸近級数展開の形として以下の形：

$$
F_0(x) = \sum_{k=0}^{n}(-1)^k k!x^{k+1} + (-1)^{n+1}F_{n+1}(x)
$$

に展開することができる。この級数は有限項と残差項の和の形になっているが、$x=0.1$の時の第k項は、


```python
def f_k(x, k): #F_0(x)を漸近級数展開した際の第k項
    return (-1.0)**k * math.factorial(k) * x**(k+1)
```


```python
y_k = np.array([f_k(0.1, i) for i in range(1,50)])
y_k
```




    array([-1.00000000e-02,  2.00000000e-03, -6.00000000e-04,  2.40000000e-04,
           -1.20000000e-04,  7.20000000e-05, -5.04000000e-05,  4.03200000e-05,
           -3.62880000e-05,  3.62880000e-05, -3.99168000e-05,  4.79001600e-05,
           -6.22702080e-05,  8.71782912e-05, -1.30767437e-04,  2.09227899e-04,
           -3.55687428e-04,  6.40237371e-04, -1.21645100e-03,  2.43290201e-03,
           -5.10909422e-03,  1.12400073e-02, -2.58520167e-02,  6.20448402e-02,
           -1.55112100e-01,  4.03291461e-01, -1.08888695e+00,  3.04888345e+00,
           -8.84176199e+00,  2.65252860e+01, -8.22283865e+01,  2.63130837e+02,
           -8.68331762e+02,  2.95232799e+03, -1.03331480e+04,  3.71993327e+04,
           -1.37637531e+05,  5.23022617e+05, -2.03978821e+06,  8.15915283e+06,
           -3.34525266e+07,  1.40500612e+08, -6.04152631e+08,  2.65827157e+09,
           -1.19622221e+10,  5.50262216e+10, -2.58623242e+11,  1.24139156e+12,
           -6.08281864e+12])



の計算でわかる通り、最初は徐々に絶対値が小さくなっていくが、10項目を超えたあたりから絶対値は大きくなっていき、次第には発散してしまう。

漸近級数展開による近似は、絶対値が最小となるような第k項までを有限級数として足し上げ、残りの残差を最小化する近似法である。これは、この展開が漸近級数であるという事実により、第$n$まで足し上げて以降を打ち切った際に残った残差$R_{n}$が、

$$
F_0(x) - \sum_{k=0}^{n}(-1)^k k!x^{k+1} \sim \mathcal{O}(n!x^{n+1}) \ \ \ \ (x \rightarrow 0)
$$

という漸近関係を持っていることが保証されることにより成立する近似である。

絶対値が最小となるような項は、


```python
print("第",np.argmin(abs(y_k))+1,"項目")
print("最小値：", '{:.3E}'.format(min(abs(y_k))))
```

    第 9 項目最小値： 3.629E-05


よって、第9(10項）目まで足し上げれば、残差を最小とするような近似になっている。


```python
y_asympt = 0.0for k in range(10):    y_asympt += f_k(0.1, k)print("近似値：", y_asympt)print("関数値：", F_0(0.1)) 
```

    近似値： 0.091545632
    関数値： 0.09156333394018187


### 1-2. 0次の第1種ベッセル関数の例

0次の第1種ベッセル関数$J_{0}(x)$はベッセルの微分方程式：

$$
x^{2}{\frac  {d^{2}y}{dx^{2}}}+x{\frac  {dy}{dx}}+x^{2}y=0
$$

の解であり、以下のような振る舞いをする関数である。


```python
from scipy.special import jv
x0 = np.array([i*0.1 for i in range(120)])
y0 = np.array([jv(0,0.1*i) for i in range(120)])
plt.plot(x0,y0,label="No Approximate")
plt.title("Bessel function(x)")
plt.xlabel("x")
plt.ylabel("J_0(x)")
plt.legend()
plt.show()
```


![output_17_0](https://user-images.githubusercontent.com/54795218/117119938-ac90be00-adcd-11eb-8f5d-3b287d08657a.png)

第1種ベッセル関数は
$x\rightarrow 0$の極限の下ではべき級数展開によって以下のように定義することもできる。

$$
\displaystyle J_0(x)=\sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2} \left(\frac{x}{2}\right)^{2m} \ \ \ (x\rightarrow 0)
$$

べき級数の第10項までを足し上げて、ベッセル関数を近似したものを以下に図示する。


```python
def J_0(x):
    sum = 0.0;
    for i in range(10):
        sum += (-x**2/4)**i / (math.factorial(i))**2
    return sum
```


```python
x1 = np.array([i*0.1 for i in range(90)])
y1 = np.array([J_0(i*0.1) for i in range(90)]) 
plt.plot(x0,y0,label="No Approximate")
plt.plot(x1,y1,label="Power-series Approximate",linestyle="dashed")
plt.legend()
plt.show()
```


![output_20_0](https://user-images.githubusercontent.com/54795218/117119944-adc1eb00-adcd-11eb-9d35-1dc28342696a.png)


このように、べき級数展開による近似は$x$が小さい値を取る領域では良い近似となるものの、$x$が大きい値を取る領域では良い近似ではなくなることがわかる。

次に、$x\rightarrow \infty$において漸近展開したものを以下に図示する。


```python
def P(x):
    sum = 0.0
    for k in range(4):
        sum += (-1)**k * math.gamma(0.5+2*k) / ( math.factorial(2*k) * math.gamma(0.5-2*k) * (2*x)**(2*k) )
    return sum

def Q(x):
    sum = 0.0
    for k in range(5):
        sum += (-1)**k * math.gamma(0.5+2*k+1.0) / ( math.factorial(2*k+1.0) * math.gamma(0.5-2*k-1.0) * (2*x)**(2*k+1.0) )
    return sum
```


```python
def J_asympt(x):
    return np.sqrt(2.0/np.pi/x) * ( P(x) * np.cos(x-np.pi/4) - Q(x) * np.sin(x-np.pi/4) ) 
```


```python
x2 = np.array([i*0.1 for i in range(15,120)])
y2 = np.array([J_asympt(i*0.1) for i in range(15,120)]) 
plt.plot(x0,y0,label="No Approximate")
plt.plot(x1,y1,label="Power-series Approximate",linestyle="dashed")
plt.plot(x2,y2,label="Asymptotic Approximate",linestyle="dashed")
plt.legend()
plt.show()
```


![output_25_0](https://user-images.githubusercontent.com/54795218/117119947-ae5a8180-adcd-11eb-8964-21fefdfaf2fc.png)


このように、漸近級数展開による近似は$x$が小さい値を取る領域では良い近似とはならないものの、$x$が大きい値を取る領域では良い近似であることがわかる。

## 2.確率過程の漸近展開

### 2-1. 幾何ブラウン運動の例

確率変数$x(t)$が以下の確率微分方程式：

$$
dx(t) = x(t)\mu dt + \epsilon x(t)\sigma dW(t)
$$

に従うものとする。この確率変数$x(t)$は漸近展開により以下の近似式：

$$
x_{\epsilon}(t) = x_0(t) + \epsilon x_1(t) + \epsilon^2 x_2(t) + \cdots
$$

が成立する。ここで、

$$
x_0(t) = x(0)e^{\mu t} \\
x_1(t) = x(0)\sigma e^{\mu t}W(t) \\
x_2(t) = x(0)\sigma e^{\mu t}\left( \frac{1}{2}W^2(t) - \frac{1}{2}t \right)
$$

である。

以下のように、伊藤の公式を用いた解析解、$\mathcal{O}(\epsilon)$までの漸近展開、$\mathcal{O}(\epsilon^2)$までの漸近展開を定義して、比較したものを図示する。


```python
def Geometric_Brownian(x_0, nu, sigma, epsilon, t, W_T):
    return x_0 * np.exp( (nu-epsilon*epsilon*sigma*sigma/2.0)*t + epsilon*sigma*W_T)

def Asymptic_expansion1(x_0, nu, sigma, epsilon, t, W_T):
    term1 = 1.0
    term2 = epsilon*sigma*W_T
    return x_0 * np.exp( nu*t ) * ( term1 + term2 )

def Asymptic_expansion2(x_0, nu, sigma, epsilon, t, W_T):
    term1 = 1.0
    term2 = epsilon*sigma*W_T
    term3 = 0.5*epsilon*epsilon*sigma*(W_T*W_T-t)
    return x_0 * np.exp( nu*t ) * ( term1 + term2 + term3 )
```


```python
dt = 1.0/ 128.0
W_t = np.zeros(128)
for i in range(1,128):
    W_t[i] = W_t[i-1] + np.sqrt(dt) * np.random.normal(0.0,1.0)
    
x_0 = 0.1;
nu = 0.5;
sigma = 5.0;
epsilon = 0.01;
t = 1.0;

x = np.array([i*dt for i in range(128)])
y0 = Geometric_Brownian(x_0, nu, sigma, epsilon, t, W_t)
y1 = Asymptic_expansion1(x_0, nu, sigma, epsilon, t, W_t)
y2 = Asymptic_expansion2(x_0, nu, sigma, epsilon, t, W_t)
```


```python
plt.plot(x, y0,label='Analytics')
plt.plot(x, y1,label='Asymptotic expansion(O(1))',linestyle="dashed")
plt.plot(x, y2,label='Asymptotic expansion(O(2))',linestyle="dashed")
plt.xlabel('time') 
plt.ylabel('x(t)') 
plt.legend()
plt.show()
```


![output_32_0](https://user-images.githubusercontent.com/54795218/117119952-aef31800-adcd-11eb-8f40-aa8d7c1e40ec.png)


このように、漸近展開で高次の項を入れていくことで、解析解に近づいていくことが分かる。

### 2-2. Fokker-Plank方程式

確率変数$x(t)$が以下の確率微分方程式：

$$
dx(t) = a(x)dt + \epsilon b(x)dW(t)
$$

に従う時、確率変数$x(t)$の確率密度関数$p(x)$は以下のように時間発展する：

$$
\partial_t p = -\partial_x[a(x)p] + \frac{1}{2}\epsilon^2\partial^2_x[b(x)^2p]
$$

ここで、初期値を中心として、$\epsilon$で規格化した変数$y$：

$$
y=[x-x_0(t)]/\epsilon\\
\bar{p}_{\epsilon}(y,t)=\epsilon p(x,t)
$$

を定義すると、$\bar{p}_{\epsilon}(y,t)$は以下のように展開することができる；

$$
\bar{p}_{\epsilon}(y,t) = \bar{p}_{0}(y,t) + \epsilon \bar{p}_{1}(y,t) + \epsilon^2 \bar{p}_{2}(y,t) + \cdots
$$


```python

```
