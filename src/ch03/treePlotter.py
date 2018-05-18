"""
create by 2018-05-18

@author: Shiyipaisizuo
"""

import matplotlib.pyplot as plt


# 定义文本框和箭头格式
decisionnode = dict(boxstyle='sawtooth', fc='0.8')
leafnode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


# 绘制带箭头的注释
def plot_node(nodetxt, centerpt, parentpt, nodetype):
    create_plot.ax1.annotate(nodetxt, xy=parentpt, xycoords='axes fraction',
                            xytext=centerpt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodetype, arrowprops=arrow_args)

# 获取叶子节点的数目
def get_num_leafs(mytree):
    numleafs = 0
    firststr = mytree.keys()[0]
    seconddict = mytree[firststr]

    # 测试节点数据是否为字典
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            numleafs += get_num_leafs(seconddict[key])
        else:
            numleafs += 1
    return numleafs


# 获取决策树的层数
def get_tree_depth(mytree):
    maxdepth = 0
    firststr = mytree.keys()[0]
    seconddict = mytree[firststr]

    # 测试数据是否为字典
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            thisdepth = 1 + get_tree_depth(seconddict[key])
        else:
            thisdepth = 1
        if thisdepth > maxdepth: maxdepth = thisdepth
    return maxdepth


# 预先储存树的信息，避免每次都要从数据中创建树的麻烦
def retrieve_tree(i):
    listoftrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listoftrees[i]


# 在父节点中填充文本的信息
def plot_mid_text(cntrpt, parentpt, txtstring):
    xmid = (parentpt[0] - cntrpt[0]) / 2.0 + cntrpt[0]
    ymid = (parentpt[1] - cntrpt[1]) / 2.0 + cntrpt[1]
    create_plot.ax1.text(xmid, ymid, txtstring, va="center", ha="center", rotation=30)


# 绘制树
def plot_tree(mytree, parentpt, nodetxt):

    # 计算树的宽和高
    numleafs = get_num_leafs(mytree)
    depth = get_tree_depth(mytree)
    firststr = mytree.keys()[0]
    cntrpt = (plot_tree.xoff + (1.0 + float(numleafs)) / 2.0 / plot_tree.totalw, plot_tree.yoff)
    plot_mid_text(cntrpt, parentpt, nodetxt)
    plot_node(firststr, cntrpt, parentpt, decisionnode)
    seconddict = mytree[firststr]
    plot_tree.yoff = plot_tree.yoff - 1.0 / plot_tree.totald

    # 测试数据是否为字典
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            plot_tree(seconddict[key], cntrpt, str(key))

        else:  # 它是叶子节点打印叶子节点
            plot_tree.xoff = plot_tree.xoff + 1.0 / plot_tree.totalw
            plot_node(seconddict[key], (plot_tree.xoff, plot_tree.yoff), cntrpt, leafnode)
            plot_mid_text((plot_tree.xoff, plot_tree.yoff), cntrpt, str(key))
    plot_tree.yoff = plot_tree.yoff + 1.0 / plot_tree.totald


# 如果你得到了一个dictonary，你知道它是一棵树，第一个元素将是另一个法令。
def create_plot(intree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    plot_tree.totalw = float(get_num_leafs(intree))
    plot_tree.totald = float(get_tree_depth(intree))
    plot_tree.xoff = -0.5 / plot_tree.totalw;
    plot_tree.yoff = 1.0;
    plot_tree(intree, (0.5, 1.0), '')
    plt.show()




