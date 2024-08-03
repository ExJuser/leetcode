package common

import (
	"sort"
)

// 83. 删除排序链表中的重复元素 保留一个
func deleteDuplicates(head *ListNode) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		if node.Val != node.Next.Val {
			node.Next = dfs(node.Next)
			return node
		} else {
			val := node.Val
			for node != nil && node.Val == val {
				node = node.Next
			}
			return &ListNode{Val: val, Next: dfs(node)}
		}
	}
	return dfs(head)
}

// 110. 平衡二叉树 左右子树高度差不超过1
func isBalanced(root *TreeNode) bool {
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		var left, right int
		if left = dfs(node.Left); left == -1 {
			return left
		}
		if right = dfs(node.Right); right == -1 {
			return right
		}
		if Abs(left-right) > 1 {
			return -1
		}
		return max(left, right) + 1
	}
	return dfs(root) != -1
}

// 404. 左叶子之和
func sumOfLeftLeaves(root *TreeNode) int {
	var sum int
	var dfs func(node *TreeNode, flag bool)
	dfs = func(node *TreeNode, flag bool) {
		if node == nil {
			return
		}
		if flag && node.Left == nil && node.Right == nil {
			sum += node.Val
		}
		dfs(node.Left, true)
		dfs(node.Right, false)
	}
	dfs(root, false)
	return sum
}

// 416. 分割等和子集 01背包问题 先遍历物品 再倒序遍历背包
func canPartition(nums []int) bool {
	var sum int
	for _, num := range nums {
		sum += num
	}
	if sum%2 != 0 {
		return false
	}
	target := sum / 2
	//dpi 容量为i的背包所能装的最大价值
	dp := make([]int, target+1)
	for i := 0; i < len(nums); i++ {
		for j := target; j >= nums[i]; j-- {
			dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])
		}
	}
	return dp[target] == target
}

// 839. 相似字符串组 并查集
func numSimilarGroups(strs []string) int {
	var isSimilar func(str1, str2 string) bool
	isSimilar = func(str1, str2 string) bool {
		var diff int
		for i := 0; i < len(str1); i++ {
			if str1[i] != str2[i] {
				diff++
			}
		}
		if diff == 0 || diff == 2 {
			return true
		}
		return false
	}
	n := len(strs)
	father := make([]int, n)
	for i := 0; i < n; i++ {
		father[i] = i
	}
	var find func(x int) int
	find = func(x int) int {
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	var union func(x, y int)
	union = func(x, y int) {
		father[find(x)] = find(y)
	}
	sets := n
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if find(i) != find(j) && isSimilar(strs[i], strs[j]) {
				union(i, j)
				sets--
			}
		}
	}
	return sets
}

// 547. 省份数量 并查集
func findCircleNum(isConnected [][]int) int {
	n := len(isConnected)
	father := make([]int, n)
	sets := n
	for i := 0; i < n; i++ {
		father[i] = i
	}
	var find func(x int) int
	find = func(x int) int {
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	var union func(x, y int)
	union = func(x, y int) {
		father[find(x)] = find(y)
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if isConnected[i][j] == 1 && find(i) != find(j) {
				union(i, j)
				sets--
			}
		}
	}
	return sets
}

// 947. 移除最多的同行或同列石头
func removeStones(stones [][]int) int {
	//记录每一行和每一列的第一块石头
	//如果遍历到的石头是一行且一列的第一块石头 不用消除
	//否则合并（随便和行合并还是列合并）
	colFirst := make(map[int]int)
	rowFirst := make(map[int]int)
	n := len(stones)
	father := make([]int, n)
	sets := n
	for i := 0; i < n; i++ {
		father[i] = i
	}
	var find func(x int) int
	find = func(x int) int {
		//如果一个节点的父结点不是自己：不是一个集合的代表节点
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	var union func(x, y int)
	union = func(x, y int) {
		//如果两个节点不属于一个集合：父结点不同
		if find(x) != find(y) {
			//任意将一个节点的父结点直接挂在另外一个节点的父节点上
			father[find(x)] = find(y)
			sets--
		}
	}
	for i, stone := range stones {
		row, col := stone[0], stone[1]
		//如果是一行的第一块石头
		if _, ok := rowFirst[row]; !ok {
			rowFirst[row] = i
		} else {
			union(rowFirst[row], i)
		}
		//如果是一列的第一块石头
		if _, ok := colFirst[col]; !ok {
			colFirst[col] = i
		} else {
			union(colFirst[col], i)
		}
	}
	return n - sets
}

// 2092. 找出知晓秘密的所有专家 并查集打上标签 标记知道秘密的集合代表元素
func findAllPeople(n int, meetings [][]int, firstPerson int) (ans []int) {
	sort.Slice(meetings, func(i, j int) bool {
		return meetings[i][2] < meetings[j][2]
	})

	father := make([]int, n)
	secret := make([]bool, n)
	for i := 0; i < n; i++ {
		father[i] = i
		secret[i] = false
	}
	father[firstPerson] = 0
	secret[0] = true

	var (
		find  func(x int) int
		union func(x, y int)
	)
	find = func(x int) int {
		if father[x] != x {
			father[x] = find(father[x])
		}
		return father[x]
	}
	union = func(x, y int) {
		fx, fy := find(x), find(y)
		if fx != fy {
			father[fx] = fy
			//如果x知道秘密 即secret[fx]=true 也应该让fy知道秘密
			//如果y知道秘密 无需设置 因为现在x已经挂在了y上面
			//核心语句
			secret[fy] = secret[fy] || secret[fx]
		}
	}

	for i := 0; i < len(meetings); {
		j := i
		//一次处理同一时刻的会议
		for j+1 < len(meetings) && meetings[j+1][2] == meetings[i][2] {
			j++
		}
		//将同一时刻的专家全都加入一组
		for l := i; l <= j; l++ {
			people1, people2 := meetings[l][0], meetings[l][1]
			union(people1, people2)
		}
		//如果这些专家开完会还是不知道秘密 初始化其指向
		for l := i; l <= j; l++ {
			people1, people2 := meetings[l][0], meetings[l][1]
			if !secret[find(people1)] {
				father[people1] = people1
			}
			if !secret[find(people2)] {
				father[people2] = people2
			}
		}
		i = j + 1
	}
	for i := 0; i < n; i++ {
		if secret[find(i)] {
			ans = append(ans, i)
		}
	}
	return
}
