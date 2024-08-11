package common

import (
	"container/heap"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
)

type minHeap [][3]int

func (m *minHeap) Len() int {
	return len(*m)
}

func (m *minHeap) Less(i, j int) bool {
	return (*m)[i][2] < (*m)[j][2]
}

func (m *minHeap) Swap(i, j int) {
	(*m)[i], (*m)[j] = (*m)[j], (*m)[i]
}

func (m *minHeap) Push(x any) {
	*m = append(*m, x.([3]int))
}

func (m *minHeap) Pop() any {
	x := (*m)[m.Len()-1]

	*m = (*m)[:m.Len()-1]
	return x
}

// 1631. 最小体力消耗路径
func minimumEffortPath(heights [][]int) int {
	cost := make([][]int, len(heights))
	visited := make([][]bool, len(heights))
	for i := 0; i < len(heights); i++ {
		cost[i] = make([]int, len(heights[i]))
		visited[i] = make([]bool, len(heights[i]))
	}
	for i := 0; i < len(cost); i++ {
		for j := 0; j < len(cost[i]); j++ {
			cost[i][j] = math.MaxInt
		}
	}
	directions := [][]int{
		{-1, 0}, {1, 0}, {0, 1}, {0, -1},
	}
	cost[0][0] = 0
	hp := &minHeap{}
	heap.Push(hp, [3]int{0, 0, 0})
	for hp.Len() > 0 {
		x := heap.Pop(hp).([3]int)
		//w是从原点到(i,j)的距离
		i, j, w := x[0], x[1], x[2]
		if i == len(heights)-1 && j == len(heights[0])-1 {
			return w
		}
		if !visited[i][j] {
			visited[i][j] = true
			for _, dir := range directions {
				ii, jj := i+dir[0], j+dir[1]
				if ii >= 0 && ii < len(heights) && jj >= 0 && jj < len(heights[0]) && !visited[ii][jj] {
					dist := Abs(heights[i][j] - heights[ii][jj])
					if max(dist, w) < cost[ii][jj] {
						cost[ii][jj] = max(dist, w)
						heap.Push(hp, [3]int{ii, jj, cost[ii][jj]})
					}
				}
			}
		}
	}
	return -1
}

// 778. 水位上升的泳池中游泳
func swimInWater(grid [][]int) int {
	cost := make([][]int, len(grid))
	visited := make([][]bool, len(grid))
	for i := 0; i < len(grid); i++ {
		cost[i] = make([]int, len(grid[0]))
		visited[i] = make([]bool, len(grid[0]))
	}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			cost[i][j] = math.MaxInt
		}
	}
	hp := &minHeap{}
	heap.Push(hp, [3]int{0, 0, 0})
	directions := [][]int{
		{0, 1}, {0, -1}, {-1, 0}, {1, 0},
	}
	for hp.Len() > 0 {
		x := heap.Pop(hp).([3]int)
		i, j, w := x[0], x[1], x[2]
		if i == len(grid)-1 && j == len(grid[0])-1 {
			return w
		}
		if !visited[i][j] {
			visited[i][j] = true
			for _, dir := range directions {
				ii, jj := i+dir[0], j+dir[1]
				if ii >= 0 && ii < len(grid) && jj >= 0 && jj < len(grid[0]) && !visited[ii][jj] {
					dist := max(w, grid[i][j], grid[ii][jj])
					if dist < cost[ii][jj] {
						cost[ii][jj] = dist
						heap.Push(hp, [3]int{ii, jj, dist})
					}
				}
			}
		}
	}
	return -1
}

// 能走的条件：不越界 不是# 没走过
// 第三个数字表示状态：如果有两把锁就是0-3 三把锁就是0-8
// 864. 获取所有钥匙的最短路径
func shortestPathAllKeys(grid []string) int {
	//先遍历一遍确定k：钥匙个数
	var k, startI, startJ int
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			//如果是小写字母代表一把钥匙
			if grid[i][j] >= 'a' && grid[i][j] <= 'z' {
				k++
			}
			if grid[i][j] == '@' {
				startI, startJ = i, j
			}
		}
	}
	target := int(math.Pow(2, float64(k))) - 1
	directions := [][]int{
		{0, 1}, {0, -1}, {1, 0}, {-1, 0},
	}
	visited := make([][][64]bool, len(grid))
	for i := 0; i < len(visited); i++ {
		visited[i] = make([][64]bool, len(grid[0]))
	}
	var moves int
	queue := make([][3]int, 0)
	//所在坐标+状态
	queue = append(queue, [3]int{startI, startJ, 0})
	for len(queue) > 0 {
		moves++
		for size := len(queue); size > 0; size-- {
			x := queue[0]
			queue = queue[1:]
			i, j, status := x[0], x[1], x[2]
			//所在位置是一把钥匙 需要设置状态位标记为钥匙已经获得
			if grid[i][j] >= 'a' && grid[i][j] <= 'z' {
				status |= 1 << (grid[i][j] - 'a')
			}
			if status == target { //拿到了所有的钥匙
				return moves - 1
			}
			visited[i][j][status] = true
			for _, dir := range directions {
				ii, jj := i+dir[0], j+dir[1] //新到达的位置
				//没有超出边界且不是墙且没有访问过
				if ii >= 0 && jj >= 0 && ii < len(grid) && jj < len(grid[0]) && grid[ii][jj] != '#' && !visited[ii][jj][status] {
					//如果遇到了锁 需要判断能否进去
					if !(grid[ii][jj] >= 'A' && grid[ii][jj] <= 'Z' && status&(1<<(grid[ii][jj]-'A')) < 1) {
						//最开始忘了这一句 会导致非常多的无效步
						visited[ii][jj][status] = true
						queue = append(queue, [3]int{ii, jj, status})
					}
				}
			}
		}
	}
	return -1
}

/**
建图：邻接表、入度map
拓扑排序
迪杰斯特拉：最小堆
克鲁斯卡尔：最小堆
分层图：图+一个状态标记是否访问过
并查集：
弗洛伊德算法
A*算法
*/

type planHeap [][3]int

func (p *planHeap) Len() int {
	return len(*p)
}

func (p *planHeap) Less(i, j int) bool {
	return (*p)[i][1] < (*p)[j][1]
}

func (p *planHeap) Swap(i, j int) {
	(*p)[i], (*p)[j] = (*p)[j], (*p)[i]
}

func (p *planHeap) Push(x any) {
	*p = append(*p, x.([3]int))
}

func (p *planHeap) Pop() any {
	x := (*p)[p.Len()-1]
	*p = (*p)[:p.Len()-1]
	return x
}

// LCP 35. 电动车游城市
func electricCarPlan(paths [][]int, cnt int, start int, end int, charge []int) int {
	//cnt 电动车最大电量 初始电量为0
	//start end 起点和终点
	//charge 单位电量充电时间 长度为城市数量
	//paths[i][j]:城市i到城市j的距离也即行驶用时
	//堆里放的东西：点 起点到该点的距离/行驶用时
	type tuple struct {
		location int
		time     int
		curCnt   int
	}

	n := len(charge)

	//建图 无向图
	graph := make([][][2]int, n)
	for _, path := range paths {
		u, v, w := path[0], path[1], path[2]
		graph[u] = append(graph[u], [2]int{v, w})
		graph[v] = append(graph[v], [2]int{u, w})
	}

	//从起点到终点的最短用时初始化
	cost := make([][]int, n)
	for i := 0; i < n; i++ {
		cost[i] = make([]int, cnt+1)
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= cnt; j++ {
			cost[i][j] = math.MaxInt
		}
	}
	cost[start][0] = 0

	//所在位置+用时+目前电量表示一个状态 地图上的一个广义点
	visited := make(map[tuple]bool)

	hp := &planHeap{} //迪杰斯特拉算法用的堆结构
	heap.Push(hp, [3]int{start, 0, 0})

	for hp.Len() > 0 {
		x := heap.Pop(hp).([3]int)
		curLocation, curTime, curCnt := x[0], x[1], x[2]
		if curLocation == end { //到达终点 返回结果
			return curTime
		}
		if !visited[tuple{curLocation, curTime, curCnt}] {
			visited[tuple{curLocation, curTime, curCnt}] = true
			//可以充电
			if curCnt < cnt {
				chargedTime := curTime + charge[curLocation]
				chargedCnt := curCnt + 1
				//判断是否访问过其实是不需要的 在出堆的时候会判断
				if cost[curLocation][chargedCnt] > chargedTime {
					cost[curLocation][chargedCnt] = chargedTime
					heap.Push(hp, [3]int{curLocation, chargedTime, chargedCnt})
				}
			}

			//不充电
			for _, to := range graph[curLocation] {
				nextLocation := to[0]
				timeCost := to[1]
				//电量足够的情况下才能去下一个城市
				if curCnt >= timeCost {
					arriveTime := curTime + timeCost
					arriveCnt := curCnt - timeCost
					if cost[nextLocation][arriveCnt] > arriveTime {
						cost[nextLocation][arriveCnt] = arriveTime
						heap.Push(hp, [3]int{nextLocation, arriveTime, arriveCnt})
					}
				}
			}
		}
	}
	return -1
}

type flightHeap [][3]int

func (m *flightHeap) Len() int {
	return len(*m)
}

func (m *flightHeap) Less(i, j int) bool {
	return (*m)[i][2] < (*m)[j][2]
}

func (m *flightHeap) Swap(i, j int) {
	(*m)[i], (*m)[j] = (*m)[j], (*m)[i]
}

func (m *flightHeap) Push(x any) {
	*m = append(*m, x.([3]int))
}

func (m *flightHeap) Pop() any {
	x := (*m)[m.Len()-1]

	*m = (*m)[:m.Len()-1]
	return x
}

// 787. K 站中转内最便宜的航班
func findCheapestPrice(n int, flights [][]int, src int, dst int, k int) int {
	//建有向图
	graph := make([][][2]int, n)
	for _, flight := range flights {
		from, to, price := flight[0], flight[1], flight[2]
		graph[from] = append(graph[from], [2]int{to, price})
	}
	//从src到达某个城市i、使用的中转次数为j的最小花费
	cost := make([][]int, n)
	for i := 0; i < n; i++ {
		cost[i] = make([]int, k+1)
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= k; j++ {
			cost[i][j] = math.MaxInt
		}
	}
	cost[src][0] = 0

	type tuple struct {
		city  int //所在城市
		times int //中转次数
		price int //当前花费
	}
	visited := make(map[tuple]bool)

	hp := &flightHeap{}
	heap.Push(hp, [3]int{src, 0, 0})
	for hp.Len() > 0 {
		x := heap.Pop(hp).([3]int)
		city, times, price := x[0], x[1], x[2]
		if city == dst {
			return price
		}
		if !visited[tuple{city, times, price}] {
			visited[tuple{city, times, price}] = true
			//选择中转城市 前提是可以中转
			for _, to := range graph[city] {
				//下一站城市 到达下一站的花费
				toCity, toPrice := to[0], to[1]
				if toCity == dst { //不算做一次中转
					if price+toPrice < cost[toCity][times] {
						cost[toCity][times] = price + toPrice
						heap.Push(hp, [3]int{toCity, times, price + toPrice})
					}
				} else { //必须不超过中转次数
					if times+1 <= k && price+toPrice < cost[toCity][times+1] {
						cost[toCity][times+1] = price + toPrice
						heap.Push(hp, [3]int{toCity, times + 1, price + toPrice})
					}
				}
			}
		}
	}
	return -1
}

// 114. 二叉树展开为链表
func flatten(root *TreeNode) {
	var dfs func(node *TreeNode) *TreeNode
	var findMostRight func(node *TreeNode) *TreeNode
	dfs = func(node *TreeNode) *TreeNode {
		if node == nil || node.Left == nil && node.Right == nil {
			return node
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		node.Right = left
		mostRight := findMostRight(node)
		mostRight.Right = right
		node.Left = nil
		return node
	}
	findMostRight = func(node *TreeNode) *TreeNode {
		for node.Right != nil {
			node = node.Right
		}
		return node
	}
	dfs(root)
}

func quickSort_(nums []int) []int {
	var helper func(left, right int)
	helper = func(left, right int) {
		if left >= right {
			return
		}
		i, j := left, right
		pivot := nums[rand.IntN(right-left+1)+left]
		for i <= j {
			for nums[i] < pivot {
				i++
			}
			for nums[j] > pivot {
				j--
			}
			if i <= j {
				nums[i], nums[j] = nums[j], nums[i]
				i++
				j--
			}
		}
		helper(left, j)
		helper(i, right)
	}
	helper(0, len(nums)-1)
	return nums
}

type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

func copyRandomList(head *Node) *Node {
	//建立原链表和新链表之间节点的映射关系
	mp := make(map[*Node]*Node)
	dummy := &Node{}
	for p, q := head, dummy; p != nil; p = p.Next {
		q.Next = &Node{
			Val: p.Val,
		}
		q = q.Next
		mp[p] = q
	}
	for p, q := head, dummy.Next; p != nil && q != nil; p, q = p.Next, q.Next {
		q.Random = mp[p.Random]
	}
	return dummy.Next
}

// 151. 反转字符串中的单词
func reverseWords(s string) string {
	//遇到非空格向后遍历 遇到空格加入slice
	//遇到空格 向后遍历直到非空格
	strs := make([]string, 0, len(s))
	for i := 0; i < len(s); {
		j := i + 1
		if s[i] == ' ' {
			for j < len(s) && s[j] == ' ' {
				j++
			}
		} else {
			for j < len(s) && s[j] != ' ' {
				j++
			}
			strs = append(strs, s[i:j])
		}
		i = j
	}
	slices.Reverse(strs)
	return strings.Join(strs, " ")
}

// 19. 删除链表的倒数第 N 个结点 快慢指针
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Next: head}
	slow, fast := dummy, dummy
	for ; n > 0; n-- {
		fast = fast.Next
	}
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}

// 54. 螺旋矩阵
func spiralOrder(matrix [][]int) (ans []int) {
	left, right := 0, len(matrix[0])-1
	top, bottom := 0, len(matrix)-1
	for left < right && top < bottom {
		for i := left; i <= right; i++ {
			ans = append(ans, matrix[top][i])
		}
		top++
		for i := top; i <= bottom; i++ {
			ans = append(ans, matrix[i][right])
		}
		right--
		for i := right; i >= left; i-- {
			ans = append(ans, matrix[bottom][i])
		}
		bottom--
		for i := bottom; i >= top; i-- {
			ans = append(ans, matrix[i][left])
		}
		left++
	}
	if left == right {
		for i := top; i <= bottom; i++ {
			ans = append(ans, matrix[i][left])
		}
	} else if top == bottom {
		for i := left; i <= right; i++ {
			ans = append(ans, matrix[top][i])
		}
	}
	return
}

//	func permute(nums []int) (ans [][]int) {
//		//不含重复数字的全排列
//		visited := make([]bool, len(nums))
//		var dfs func(path []int)
//		dfs = func(path []int) {
//			if len(path) == len(nums) {
//				ans = append(ans, append([]int{}, path...))
//				return
//			}
//			for i := 0; i < len(nums); i++ {
//				if !visited[i] {
//					visited[i] = true
//					path = append(path, nums[i])
//					dfs(path)
//					visited[i] = false
//					path = path[:len(path)-1]
//				}
//			}
//		}
//		dfs([]int{})
//		return
//	}
//
// 最长回文子串
func longestPalindrome(s string) string {
	dp := make([][]bool, len(s))
	for i := 0; i < len(s); i++ {
		dp[i] = make([]bool, len(s))
	}
	for i := 0; i < len(s); i++ {
		dp[i][i] = true
	}
	ans := s[0:1]
	for i := len(s) - 1; i >= 0; i-- {
		for j := i; j < len(s); j++ {
			if s[i] == s[j] {
				if j-i <= 1 || dp[i+1][j-1] {
					dp[i][j] = true
					if j-i+1 > len(ans) {
						ans = s[i : j+1]
					}
				}
			}
		}
	}
	return ans
}

// 3. 无重复字符的最长子串
func lengthOfLongestSubstring(s string) int {
	var left, ans int
	mp := map[byte]int{}
	for right := 0; right < len(s); right++ {
		mp[s[right]]++
		for ; len(mp) < right-left+1; left++ {
			mp[s[left]]--
			if mp[s[left]] == 0 {
				delete(mp, s[left])
			}
		}
		ans = max(ans, right-left+1)
	}
	return ans
}

// 迭代
func reverseList(head *ListNode) *ListNode {
	var pre *ListNode
	for cur := head; cur != nil; {
		nxt := cur.Next
		cur.Next = pre
		cur, pre = nxt, cur
	}
	return pre
}

func reverseListRecursive(head *ListNode) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		newHead := dfs(node.Next)
		node.Next.Next = node
		node.Next = nil
		return newHead
	}
	return dfs(head)
}

func restoreIpAddresses(s string) (ans []string) {
	var check func(left, right int) bool
	//check直接传入left和right比较好
	check = func(left, right int) bool {
		if left > right || right >= len(s) {
			return false
		}
		str := s[left : right+1]
		if str[0] == '0' {
			return len(str) == 1
		}
		num, _ := strconv.Atoi(str)
		return num <= 255 && num >= 0
	}
	var dfs func(left, right int, path []string)
	dfs = func(left, right int, path []string) {
		if right == len(s) && len(path) == 4 {
			ans = append(ans, strings.Join(path, "."))
			return
		}

		//选择不切割 只有right-left满足条件的情况下才可以不切割
		//而且right不能为最后一位
		if right-left <= 1 && right != len(s)-1 {
			dfs(left, right+1, path)
		}

		//选择切割 此时已经切割的份数不能超过3
		if check(left, right) && len(path) <= 3 {
			path = append(path, s[left:right+1])
			dfs(right+1, right+1, path)
			path = path[:len(path)-1]
		}
	}
	dfs(0, 0, []string{})
	return
}

func mySqrt(x int) int {
	left, right := 0, x
	var ans int
	for left <= right {
		mid := (right-left)/2 + left
		res := mid * mid
		if res <= x {
			ans = mid
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return ans
}

// 557. 反转字符串中的单词 III
func reverseWords3(s string) string {
	bytes := []byte(s)
	for i := 0; i < len(bytes); {
		if bytes[i] != ' ' {
			j := i + 1
			for j < len(bytes) && bytes[j] != ' ' {
				j++
			}
			slices.Reverse(bytes[i:j])
			i = j + 1
		}
	}
	return string(bytes)
}

// 64. 最小路径和 动态规划
func minPathSum(grid [][]int) int {
	dp := make([][]int, len(grid))
	for i := 0; i < len(grid); i++ {
		dp[i] = make([]int, len(grid[i]))
	}
	dp[0][0] = grid[0][0]
	for i := 1; i < len(dp); i++ {
		dp[i][0] = dp[i-1][0] + grid[i][0]
	}
	for i := 1; i < len(grid[0]); i++ {
		dp[0][i] += dp[0][i-1] + grid[0][i]
	}
	for i := 1; i < len(grid); i++ {
		for j := 1; j < len(grid[i]); j++ {
			dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
		}
	}
	return dp[len(grid)-1][len(grid[0])-1]
}

// 1312. 让字符串成为回文串的最少插入次数 区间dp
// 1312. 让字符串成为回文串的最少插入次数
func minInsertions(s string) int {
	n := len(s)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
	}
	for i := n - 1; i >= 0; i-- {
		for j := i; j < n; j++ {
			if i == j {
				dp[i][j] = 0
			} else if i+1 == j {
				if s[i] == s[j] {
					dp[i][j] = 0
				} else {
					dp[i][j] = 1
				}
			} else {
				if s[i] == s[j] {
					dp[i][j] = dp[i+1][j-1]
				} else {
					dp[i][j] = min(dp[i+1][j], dp[i][j-1]) + 1
				}
			}
		}
	}
	return dp[0][n-1]
}

// 516. 最长回文子序列
func longestPalindromeSubseq(s string) int {
	n := len(s)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
	}
	for i := n - 1; i >= 0; i-- {
		for j := i; j < n; j++ {
			if i == j {
				dp[i][j] = 1
			} else if i+1 == j {
				if s[i] == s[j] {
					dp[i][j] = 2
				} else {
					dp[i][j] = 1
				}
			} else {
				if s[i] == s[j] {
					dp[i][j] = dp[i+1][j-1] + 2
				} else {
					dp[i][j] = max(dp[i+1][j], dp[i][j-1])
				}
			}
		}
	}
	return dp[0][n-1]
}

// 486. 预测赢家 博弈论 每个人都会做出让对方最差的选择
func predictTheWinner(nums []int) bool {
	n := len(nums)
	dp := make([][]int, n)
	var sum int
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
		sum += nums[i]
	}
	for i := n - 1; i >= 0; i-- {
		for j := i; j < n; j++ {
			if i == j {
				dp[i][j] = nums[i]
			} else if i+1 == j {
				dp[i][j] = max(nums[i], nums[j])
			} else {
				dp[i][j] = max(nums[i]+min(dp[i+2][j], dp[i+1][j-1]), nums[j]+min(dp[i][j-2], dp[i+1][j-1]))
			}
		}
	}
	return dp[0][n-1] >= sum-dp[0][n-1]
}

// 877. 石子游戏 博弈论 每个人都会做出让对方最差的选择
// 存在数学解法
func stoneGame(nums []int) bool {
	n := len(nums)
	dp := make([][]int, n)
	var sum int
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
		sum += nums[i]
	}
	for i := n - 1; i >= 0; i-- {
		for j := i; j < n; j++ {
			if i == j {
				dp[i][j] = nums[i]
			} else if i+1 == j {
				dp[i][j] = max(nums[i], nums[j])
			} else {
				dp[i][j] = max(nums[i]+min(dp[i+2][j], dp[i+1][j-1]), nums[j]+min(dp[i][j-2], dp[i+1][j-1]))
			}
		}
	}
	return dp[0][n-1] >= sum-dp[0][n-1]
}

// 1373. 二叉搜索子树的最大键值和 树形dp
// 综合子节点的信息 向上返回 可能需要返回很多信息
func maxSumBST(root *TreeNode) int {
	//左子树是BST而且右子树是BST 而且满足值的关系 那么当前子树就是二叉搜索子树
	//因此需要返回 isBST max min sum
	//空节点返回 true math.min math.max 0
	var ans int
	var dfs func(node *TreeNode) (isBST bool, max_, min_, ans int)
	dfs = func(node *TreeNode) (isBST bool, max_, min_, sum int) {
		if node == nil {
			return true, math.MinInt, math.MaxInt, 0
		}
		isLeftBST, leftMax, leftMin, leftSum := dfs(node.Left)
		isRightBST, rightMax, rightMin, rightSum := dfs(node.Right)
		max_ = max(leftMax, rightMax, node.Val)
		min_ = min(leftMin, rightMin, node.Val)
		isBST = isLeftBST && isRightBST && (node.Val > leftMax && node.Val < rightMin)
		if isBST {
			sum = leftSum + rightSum + node.Val
		} else {
			sum = max(leftSum, rightSum)
		}
		ans = max(ans, sum)
		return
	}
	dfs(root)
	return ans
}

// 543. 二叉树的直径 最简单的树形dp
func diameterOfBinaryTree(root *TreeNode) int {
	var dfs func(node *TreeNode) int
	var ans int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left, right := dfs(node.Left), dfs(node.Right)
		ans = max(ans, left+right)
		return max(left, right) + 1
	}
	dfs(root)
	return ans
}

// 979. 在二叉树中分配硬币
// 只能解决一枚硬币占一次移动次数的题目
func distributeCoins(root *TreeNode) int {
	//一个父节点需要负责将左右子树的硬币全变成1
	//如果子树的硬币大于1 那么要给父结点
	//如果子树硬币为0 要从父结点拿走 一个
	var dfs func(node *TreeNode)
	var ans int
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		dfs(node.Right)
		if node.Left != nil {
			node.Val += node.Left.Val - 1
			ans += Abs(node.Left.Val - 1)
		}
		if node.Right != nil {
			node.Val += node.Right.Val - 1
			ans += Abs(node.Right.Val - 1)
		}
	}
	dfs(root)
	return ans
}

// 337. 打家劫舍 III
func rob(root *TreeNode) int {
	//左右子树偷了任意一个就不能偷
	//返回偷当前节点和不偷当前节点的值
	var dfs func(node *TreeNode) (int, int)
	dfs = func(node *TreeNode) (int, int) {
		if node == nil {
			return 0, 0
		}
		leftRob, leftNotRob := dfs(node.Left)
		rightRob, rightNotRob := dfs(node.Right)
		robbed := leftNotRob + rightNotRob + node.Val
		notRobbed := max(leftRob, leftNotRob) + max(rightRob, rightNotRob)
		return robbed, notRobbed
	}
	a, b := dfs(root)
	return max(a, b)
}

// 968. 监控二叉树
func minCameraCover(root *TreeNode) int {
	//向上返回自己有没有放置摄像头，和有没有被监控到，当前子树的摄像头数目
	var dfs func(node *TreeNode) (bool, bool, int)
	dfs = func(node *TreeNode) (bool, bool, int) {
		if node == nil {
			return false, true, 0
		}
		leftCamera, leftCovered, leftCameraCount := dfs(node.Left)
		rightCamera, rightCovered, rightCameraCount := dfs(node.Right)
		//如果左右有一个没有被监控 那么当前节点必须放摄像头
		if !leftCovered || !rightCovered {
			return true, true, leftCameraCount + rightCameraCount + 1
		}
		//如果都被监控了 但是都没有放摄像头 由上层来放
		if !leftCamera && !rightCamera {
			return false, false, leftCameraCount + rightCameraCount
		} else { //有一个放了摄像头 当前节点就被监控到 同样无需放摄像头
			return false, true, leftCameraCount + rightCameraCount
		}
	}
	_, covered, ans := dfs(root)
	if !covered {
		return ans + 1
	}
	return ans
}

// 2477. 到达首都的最少油耗 拓扑排序
func minimumFuelCost(roads [][]int, seats int) int64 {
	//n个城市
	n := len(roads) + 1
	graph := make([]map[int]struct{}, n)
	weights := make([]int, n)
	for i := 0; i < n; i++ {
		graph[i] = make(map[int]struct{})
		weights[i] = 1
	}
	inDegrees := make(map[int]int)
	for _, road := range roads {
		//双向路 无向图
		from, to := road[0], road[1]
		graph[from][to] = struct{}{}
		graph[to][from] = struct{}{}
		//因此两个城市的入度都++
		inDegrees[from]++
		inDegrees[to]++
	}
	//入度为1的先入队列 但是首都不能加入
	queue := make([]int, 0, n)
	for k, v := range inDegrees {
		if v == 1 && k != 0 {
			queue = append(queue, k)
		}
	}
	var cost int64
	for len(queue) > 0 {
		//出队
		city := queue[0]
		queue = queue[1:]
		//找到这座城市所联通的城市 其实只有一座城市
		var toCity int
		for c, _ := range graph[city] {
			toCity = c
		}
		//向上取整
		cost += int64((weights[city] + seats - 1) / seats)
		weights[toCity] += weights[city]
		inDegrees[toCity]--
		delete(graph[city], toCity)
		delete(graph[toCity], city)
		if inDegrees[toCity] == 1 && toCity != 0 {
			queue = append(queue, toCity)
		}
	}
	return cost
}

// 2246. 相邻字符不同的最长路径 拓扑排序写成的屎山一步一步优化到树形dp
func longestPath(parent []int, s string) int {
	n := len(parent)
	graph := make([][]int, n)
	for from := 1; from < n; from++ {
		to := parent[from]
		graph[to] = append(graph[to], from)
	}
	var dfs func(i int) int
	ans := 1
	dfs = func(i int) int {
		//递归出口：叶子结点
		if len(graph[i]) == 0 {
			return 1
		}
		var max1, max2 int
		for _, k := range graph[i] {
			dfsk := dfs(k)
			//相邻元素不同
			if s[k] != s[i] {
				if dfsk > max1 {
					max2 = max1
					max1 = dfsk
				} else if dfsk > max2 {
					max2 = dfsk
				}
			}
		}
		ans = max(ans, max1+max2+1)
		return max1 + 1
	}
	dfs(0)
	return ans
}
