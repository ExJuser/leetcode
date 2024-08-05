package common

import (
	"container/heap"
	"math"
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

// 能走的条件：前两个数字不越界 不是# 没走过
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
						visited[ii][jj][status] = true
						queue = append(queue, [3]int{ii, jj, status})
					}
				}
			}
		}
	}
	return -1
}
