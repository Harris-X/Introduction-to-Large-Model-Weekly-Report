# Latex的使用

> - 入门视频：<https://www.bilibili.com/video/BV1Mc411S75c>
> - 视频：<https://www.bilibili.com/video/BV1no4y1U7At>
> - [latex中文教程-15集从入门到精通包含各种latex操作](https://www.bilibili.com/video/BV15x411j7k6)
> - [Latex公式快速文档](https://mp.weixin.qq.com/s?__biz=MzAwOTM2NjU3MQ==&mid=2652275142&idx=4&sn=396dc796bc318634bd51db491aa22467&chksm=80829c82b7f5159497c07b4dd23df862ae240512b374fe33885dc67beab5430a8829c436f1c8&scene=27)
> - [Markdown/LaTeX 数学公式和符号表](https://zhuanlan.zhihu.com/p/450465546)

## 数学公式

- [多行公式的处理](https://www.bilibili.com/video/BV15x411j7k6?p=12&vd_source=29520f96e7e37ed65f945d56966cc4db)

$$
\begin{gather*}\label{equa2}
	&\alpha+\beta = \theta \notag \\
	&a+b=c
\end{gather*}
$$

$$
\begin{equation}
	\begin{split}
		\alpha+\beta &= \theta \\
		a+b&=c \\
		a &= \text{THU}^\text{E-ACT} \times 5
	\end{split}
\end{equation}
$$

$$
\begin{equation}
	\begin{cases}
		0&,\text{if}\qquad\alpha+\beta = \theta \\
		x+1&,\text{if}\qquad a+b=c \\
		1&,\text{if}\qquad a = \text{THU}^\text{E-ACT} \times 5
	\end{cases}
\end{equation}
$$

对齐符号`&`