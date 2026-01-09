# Morphological Open Close on Small Region

每个Label对应二值蒙版的**前景**区域的小连通域蒙版（孤岛）：17个Label对应17个二值蒙版
$$
SC^i_{m+} = \operatorname{SmallComp_{<10mm^2}} \left(mask_i\right),~ i \in \{0,1,...,16\}
$$
每个Label对应二值蒙版的**背景**区域的小连通域蒙版（孔洞）：
$$
SC^i_{m-} = \operatorname{SmallComp_{<10mm^2}} \left(\neg mask_i\right),~ i \in \{0,1,...,16\}
$$
每个Label对应二值蒙版的非小连通域蒙版（不修改这些大型区域）：
$$
SC^i_{res} = \neg(SC^i_{m+} ~|~ SC^i_{m-})
$$
使用5×5结构元进行形态学开运算和闭运算。保留大型区域，形态学开运算（抑制孤岛）后的**前景**，形态学闭运算（补洞）后的**背景**：
$$
\operatorname{Optimize} \left(mask_i\right) = SC^i_{res} \odot mask_i + SC^i_{m+} \odot \operatorname{Open_{5×5}} \left(mask_i\right) + SC^i_{m-} \odot \operatorname{Close_{5×5}} \left(mask_i\right)
$$
每个单点 $(x,y,z)$ 位置在17个Label上的标签值序列：
$$
\operatorname{Idx} \left(x,y,z\right) = \left[mask_0\left(x,y,z\right),mask_1\left(x,y,z\right),...,mask_{16}\left(x,y,z\right)\right]
$$
17个处理后的二值蒙版合成1个多值蒙版，点值为对应Label的编号0~17：

- 如果标签值序列是one-hot的，那么值为1的分量对应的索引就是蒙版值；
- 如果标签不是one-hot的，有2种情况：
  - 标签值序列是0向量，说明经过处理后该点失去了全部标签，计算全部二值蒙版7×7范围内的前景点数（支持点），取支持点最多的Label值填入该点（相当于取众数）；
  - 标签值序列存在多个1分量，说明经过处理后该点的若干Label存在重叠（多义性），计算该点值为1的二值蒙版7×7范围内的前景点数（支持点），取支持点最多的Label值填入该点。

$$
mask\left(x,y,z\right) = \begin{cases}
   \operatorname{Argmax}\left(\operatorname{Idx} \left(x,y,z\right)\right) &\text{if } \operatorname{Idx} \left(x,y,z\right) \text{ is one-hot} \\
   \operatorname{Argmax}\left(\left[\operatorname{Sum^{7×7}_0}\left(x,y,z\right),\operatorname{Sum^{7×7}_1}\left(x,y,z\right),...,\operatorname{Sum^{7×7}_{16}}\left(x,y,z\right)\right]\right) &\text{if } \operatorname{Idx} \left(x,y,z\right) \text{ is all-zero} \\
   \operatorname{Argmax}\left(\left[\operatorname{Sum^{7×7}_0}\left(x,y,z\right),\operatorname{Sum^{7×7}_1}\left(x,y,z\right),...,\operatorname{Sum^{7×7}_{16}}\left(x,y,z\right)\right] \odot \operatorname{Idx} \left(x,y,z\right)\right) &\text{otherwise (multiple 1 comp.)} \\
   \end{cases}
$$

