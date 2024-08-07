package common

import (
	"container/heap"
	"math"
	"math/rand/v2"
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
				//如果在这个城市、当前时间、当前电量的状态之前没有访问过
				//if !visited[tuple{curLocation, chargedTime, chargedCnt}] {
				//	if cost[curLocation][chargedCnt] > chargedTime {
				//		cost[curLocation][chargedCnt] = chargedTime
				//		heap.Push(hp, [3]int{curLocation, chargedTime, chargedCnt})
				//	}
				//}
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
					//if !visited[tuple{nextLocation, arriveTime, arriveCnt}] {
					//	if cost[nextLocation][arriveCnt] > arriveTime {
					//		cost[nextLocation][arriveCnt] = arriveTime
					//		heap.Push(hp, [3]int{nextLocation, arriveTime, arriveCnt})
					//	}
					//}
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
