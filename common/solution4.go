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
