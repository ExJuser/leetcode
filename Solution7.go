package main

import (
	"slices"
	"strconv"
)

func countGood(nums []int, k int) (ans int64) {
	var left, pairCount int
	cnt := make(map[int]int)
	for right := 0; right < len(nums); right++ {
		pairCount += cnt[nums[right]]
		cnt[nums[right]]++
		for ; pairCount >= k; left++ {
			cnt[nums[left]]--
			pairCount -= cnt[nums[left]]
			ans += int64(len(nums) - right)
		}
	}
	return
}

// 窗口内的最大最小值的差不大于2
// 单调队列维护最大最小值
func continuousSubarrays(nums []int) (ans int64) {
	maxQueue := make([]int, 0, len(nums))
	minQueue := make([]int, 0, len(nums))
	var left int
	for right := 0; right < len(nums); right++ {
		for len(maxQueue) > 0 && maxQueue[len(maxQueue)-1] < nums[right] {
			maxQueue = maxQueue[:len(maxQueue)-1]
		}
		for len(minQueue) > 0 && minQueue[len(minQueue)-1] > nums[right] {
			minQueue = minQueue[:len(minQueue)-1]
		}
		minQueue = append(minQueue, nums[right])
		maxQueue = append(maxQueue, nums[right])
		for ; len(maxQueue) > 0 && len(minQueue) > 0 && maxQueue[0]-minQueue[0] > 2; left++ {
			if maxQueue[0] == nums[left] {
				maxQueue = maxQueue[1:]
			}
			if minQueue[0] == nums[left] {
				minQueue = minQueue[1:]
			}
		}
		ans += int64(right - left + 1)
	}
	return
}

// 回溯复习：电话号码的字母组合
func letterCombinations1(digits string) (ans []string) {
	mapping := []string{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}
	var dfs func(index int, path []byte)
	dfs = func(index int, path []byte) {
		if index == len(digits) {
			ans = append(ans, string(path))
			return
		}
		letters := mapping[digits[index]-'0']
		//遍历可能的选择并恢复现场
		for _, ch := range letters {
			path = append(path, byte(ch))
			dfs(index+1, path)
			path = path[:len(path)-1]
		}
	}
	if len(digits) == 0 {
		return
	}
	dfs(0, []byte{})
	return
}

// 左括号的出现次数永远大于等于右括号
func generateParenthesis(n int) (ans []string) {
	var dfs func(left, right int, path []byte)
	dfs = func(left, right int, path []byte) {
		if left == n && right == n {
			ans = append(ans, string(path))
			return
		}
		//添加一个左括号
		if left < n {
			path = append(path, '(')
			dfs(left+1, right, path)
			path = path[:len(path)-1]
		}
		//添加一个右括号
		if right < n && right < left {
			path = append(path, ')')
			dfs(left, right+1, path)
			path = path[:len(path)-1]
		}
	}
	dfs(0, 0, []byte{})
	return
}
func combinationSum_(candidates []int, target int) (ans [][]int) {
	var dfs func(index, sum int, path []int)
	dfs = func(index, sum int, path []int) {
		if index == len(candidates) || sum >= target {
			if sum == target {
				ans = append(ans, append([]int{}, path...))
			}
			return
		}
		//选
		path = append(path, candidates[index])
		dfs(index, sum+candidates[index], path)
		path = path[:len(path)-1]

		//不选
		dfs(index+1, sum, path)
	}
	dfs(0, 0, []int{})
	return
}
func combinationSum2_(candidates []int, target int) (ans [][]int) {
	var dfs func(index, sum int, path []int)
	dfs = func(index, sum int, path []int) {
		if index == len(candidates) || sum >= target {
			if sum == target {
				ans = append(ans, append([]int{}, path...))
			}
			return
		}
		for i := index; i < len(candidates); i++ {
			//去重操作
			if i == index || candidates[i] != candidates[i-1] {
				path = append(path, candidates[i])
				dfs(i+1, sum+candidates[i], path)
				path = path[:len(path)-1]
			}
		}
	}
	slices.Sort(candidates)
	dfs(0, 0, []int{})
	return
}

// 全排列
func permute_(nums []int) (ans [][]int) {
	visited := make([]bool, len(nums))
	var dfs func(cnt int, path []int)
	dfs = func(cnt int, path []int) {
		if cnt == len(nums) {
			ans = append(ans, append([]int{}, path...))
			return
		}
		for i, num := range nums {
			if !visited[i] {
				visited[i] = true
				path = append(path, num)
				dfs(cnt+1, path)
				visited[i] = false
				path = path[:len(path)-1]
			}
		}
	}
	dfs(0, []int{})
	return
}
func permuteUnique_(nums []int) (ans [][]int) {
	visited := make([]bool, len(nums))
	slices.Sort(nums)
	var dfs func(cnt int, path []int)
	dfs = func(cnt int, path []int) {
		if cnt == len(nums) {
			ans = append(ans, append([]int{}, path...))
			return
		}
		for i, num := range nums {
			//去重
			if !visited[i] && (i == 0 || num != nums[i-1] || visited[i-1]) {
				visited[i] = true
				path = append(path, num)
				dfs(cnt+1, path)
				visited[i] = false
				path = path[:len(path)-1]
			}
		}
	}
	dfs(0, []int{})
	return
}
func subsets_(nums []int) (ans [][]int) {
	var dfs func(index int, path []int)
	dfs = func(index int, path []int) {
		if index == len(nums) {
			ans = append(ans, append([]int{}, path...))
			return
		}
		//选
		path = append(path, nums[index])
		dfs(index+1, path)
		path = path[:len(path)-1]
		//不选
		dfs(index+1, path)
	}
	dfs(0, []int{})
	return
}
func exist(board [][]byte, word string) bool {
	used := make([][]bool, len(board))
	for i := 0; i < len(used); i++ {
		used[i] = make([]bool, len(board[i]))
	}
	var dfs func(i, j int, path []byte) bool
	dfs = func(i, j int, path []byte) bool {
		if len(path) == len(word) {
			return string(path) == word
		}
		if i+1 < len(board) && board[i+1][j] == word[len(path)] && !used[i+1][j] {
			path = append(path, board[i+1][j])
			used[i+1][j] = true
			if dfs(i+1, j, path) {
				return true
			}
			path = path[:len(path)-1]
			used[i+1][j] = false
		}
		if i >= 1 && i-1 < len(board) && board[i-1][j] == word[len(path)] && !used[i-1][j] {
			path = append(path, board[i-1][j])
			used[i-1][j] = true
			if dfs(i-1, j, path) {
				return true
			}
			path = path[:len(path)-1]
			used[i-1][j] = false
		}
		if j+1 < len(board[i]) && board[i][j+1] == word[len(path)] && !used[i][j+1] {
			path = append(path, board[i][j+1])
			used[i][j+1] = true
			if dfs(i, j+1, path) {
				return true
			}
			path = path[:len(path)-1]
			used[i][j+1] = false
		}
		if j >= 1 && j-1 < len(board[i]) && board[i][j-1] == word[len(path)] && !used[i][j-1] {
			used[i][j-1] = true
			path = append(path, board[i][j-1])
			if dfs(i, j-1, path) {
				return true
			}
			path = path[:len(path)-1]
			used[i][j-1] = false
		}
		return false
	}
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {
			if board[i][j] == word[0] {
				used[i][j] = true
				if dfs(i, j, []byte{word[0]}) {
					return true
				}
				used[i][j] = false
			}
		}
	}
	return false
}
func buildArray(target []int, n int) (ans []string) {
	list := make([]int, n)
	for i := 0; i < n; i++ {
		list[i] = i + 1
	}
	for i, j := 0, 0; i < len(target); i++ {
		if target[i] == list[j] {
			ans = append(ans, "Push")
			j++
		} else {
			cnt := 0
			for target[i] != list[j] {
				ans = append(ans, "Push")
				cnt++
				j++
			}
			for ; cnt > 0; cnt-- {
				ans = append(ans, "Pop")
			}
		}
	}
	return
}
func backspaceCompare(s string, t string) bool {
	stack1 := make([]byte, 0, len(s))
	stack2 := make([]byte, 0, len(t))
	for _, ch := range s {
		if ch != '#' {
			stack1 = append(stack1, byte(ch))
		} else {
			if len(stack1) > 0 {
				stack1 = stack1[:len(stack1)-1]
			}
		}
	}
	for _, ch := range t {
		if ch != '#' {
			stack2 = append(stack2, byte(ch))
		} else {
			if len(stack2) > 0 {
				stack2 = stack2[:len(stack2)-1]
			}
		}
	}
	return string(stack1) == string(stack2)
}

func calPoints(operations []string) int {
	stack := make([]int, 0, len(operations))
	for _, op := range operations {
		if op == "C" {
			stack = stack[:len(stack)-1]
		} else if op == "D" {
			prev := stack[len(stack)-1]
			stack = append(stack, prev*2)
		} else if op == "+" {
			prev1, prev2 := stack[len(stack)-1], stack[len(stack)-2]
			stack = append(stack, prev1+prev2)
		} else {
			score, _ := strconv.Atoi(op)
			stack = append(stack, score)
		}
	}
	scores := 0
	for _, score := range stack {
		scores += score
	}
	return scores
}

type BrowserHistory struct {
	index   int
	history []string
}

//func Constructor(homepage string) BrowserHistory {
//	return BrowserHistory{
//		index:   0,
//		history: []string{homepage},
//	}
//}

func (this *BrowserHistory) Visit(url string) {
	this.history = append(this.history[:this.index+1], url)
	this.index = len(this.history) - 1
}

func (this *BrowserHistory) Back(steps int) string {
	this.index = max(this.index-steps, 0)
	return this.history[this.index]
}

func (this *BrowserHistory) Forward(steps int) string {
	this.index = min(len(this.history)-1, this.index+steps)
	return this.history[this.index]
}

func numberOfPoints(nums [][]int) (ans int) {
	slices.SortFunc(nums, func(a, b []int) int {
		return a[0] - b[0]
	})
	maxRight := 0
	for _, num := range nums {
		if num[0] > maxRight {
			ans += num[1] - num[0] + 1
		} else {
			ans += max(0, num[1]-maxRight)
		}
		maxRight = max(maxRight, num[1])
	}
	return ans
}
