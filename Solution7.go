package main

import (
	"bytes"
	"container/heap"
	"slices"
	"strconv"
	"strings"
	"unicode"
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

func validateStackSequences(pushed []int, popped []int) bool {
	stack := make([]int, 0, len(pushed))
	pushIndex := 0
	for i := 0; i < len(popped); i++ {
		for len(stack) == 0 || stack[len(stack)-1] != popped[i] {
			if pushIndex >= len(pushed) {
				return false
			}
			stack = append(stack, pushed[pushIndex])
			pushIndex++
		}
		stack = stack[:len(stack)-1]
	}
	return true
}
func simplifyPath(path string) string {
	splits := strings.Split(path, "/")
	stack := make([]string, 0, len(path))
	for _, split := range splits {
		if split == "." || split == "" {
			continue
		} else if split == ".." {
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
		} else {
			stack = append(stack, split)
		}
	}
	return "/" + strings.Join(stack, "/")
}

//type MinStack struct {
//	stack, minStack []int
//}

//	func (this *MinStack) Push(val int) {
//		this.stack = append(this.stack, val)
//		if len(this.minStack) == 0 || this.minStack[len(this.minStack)-1] >= val {
//			this.minStack = append(this.minStack, val)
//		}
//	}
//
//	func (this *MinStack) Pop() {
//		if this.minStack[len(this.minStack)-1] == this.stack[len(this.stack)-1] {
//			this.minStack = this.minStack[:len(this.minStack)-1]
//		}
//		this.stack = this.stack[:len(this.stack)-1]
//	}
//
//	func (this *MinStack) Top() int {
//		return this.stack[len(this.stack)-1]
//	}
//
//	func (this *MinStack) GetMin() int {
//		return this.minStack[len(this.minStack)-1]
//	}
type CustomStack struct {
	stack []int
}

//func Constructor(maxSize int) CustomStack {
//	return CustomStack{stack: make([]int, 0, maxSize)}
//}

func (this *CustomStack) Push(x int) {
	if len(this.stack) < cap(this.stack) {
		this.stack = append(this.stack, x)
	}
}

func (this *CustomStack) Pop() int {
	if len(this.stack) == 0 {
		return -1
	}
	x := this.stack[len(this.stack)-1]
	this.stack = this.stack[:len(this.stack)-1]
	return x
}

func (this *CustomStack) Increment(k int, val int) {
	for i := 0; i < min(k, len(this.stack)); i++ {
		this.stack[i] += val
	}
}
func exclusiveTime(n int, logs []string) []int {
	ans := make([]int, n)
	stack := make([]int, 0, n)
	var cur, prev []string
	for i, log := range logs {
		cur = strings.Split(log, ":")
		if i != 0 {
			startTime, _ := strconv.Atoi(prev[2])
			endTime, _ := strconv.Atoi(cur[2])
			if cur[1] == "start" {
				if len(stack) > 0 {
					ans[stack[len(stack)-1]] += endTime - startTime
					if prev[1] == "end" {
						ans[stack[len(stack)-1]] -= 1
					}
				}
				funcID, _ := strconv.Atoi(cur[0])
				stack = append(stack, funcID)
			} else {
				ans[stack[len(stack)-1]] += endTime - startTime
				if prev[1] == "start" {
					ans[stack[len(stack)-1]] += 1
				}
				stack = stack[:len(stack)-1]
			}
		} else {
			funcID, _ := strconv.Atoi(cur[0])
			stack = append(stack, funcID)
		}
		prev = cur
	}
	return ans
}

// 即给定入栈顺序 求字典序最小的出栈顺序
// 若栈为空 入栈
// 若当前栈顶的字母比剩余未入栈的字母都小 出栈
// 若当前栈顶的字母不是最小 一直入栈 直到栈顶元素最小然后出栈
// 若当前栈顶的字母不是最小 但是已经没有元素可以入栈 直接出栈
// 如何判断栈顶元素和未入栈的元素之间的大小？
func robotWithString(s string) string {
	suffix := make([]byte, len(s))
	for i := len(s) - 1; i >= 0; i-- {
		if i == len(s)-1 {
			suffix[i] = s[i]
		} else {
			suffix[i] = min(suffix[i+1], s[i])
		}
	}
	ans := make([]byte, 0, len(s))
	stack := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		for len(stack) > 0 && stack[len(stack)-1] <= suffix[i] {
			ans = append(ans, stack[len(stack)-1])
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, s[i])
	}
	slices.Reverse(stack)
	ans = append(ans, stack...)
	return string(ans)
}

type FreqHeap [][3]int

func (f *FreqHeap) Len() int { return len(*f) }

func (f *FreqHeap) Less(i, j int) bool {
	if (*f)[i][1] == (*f)[j][1] {
		return (*f)[i][2] > (*f)[j][2]
	}
	return (*f)[i][1] > (*f)[j][1]
}

func (f *FreqHeap) Swap(i, j int) { (*f)[i], (*f)[j] = (*f)[j], (*f)[i] }

func (f *FreqHeap) Push(x any) { *f = append(*f, x.([3]int)) }

func (f *FreqHeap) Pop() any {
	x := (*f)[len(*f)-1]
	*f = (*f)[:len(*f)-1]
	return x
}

type FreqStack struct {
	freqHeap *FreqHeap
	cnt      map[int]int
	index    int
}

//func Constructor() FreqStack {
//	return FreqStack{
//		freqHeap: &FreqHeap{},
//		cnt:      make(map[int]int),
//	}
//}

func (this *FreqStack) Push(val int) {
	this.cnt[val]++
	this.index++
	heap.Push(this.freqHeap, [3]int{val, this.cnt[val], this.index})
}

func (this *FreqStack) Pop() int {
	num := heap.Pop(this.freqHeap).([3]int)[0]
	this.cnt[num]--
	return num
}

func minLength(s string) int {
	stack := make([]byte, 0, len(s))
	for _, ch := range s {
		if ch == 'B' && len(stack) > 0 && stack[len(stack)-1] == 'A' {
			stack = stack[:len(stack)-1]
		} else if ch == 'D' && len(stack) > 0 && stack[len(stack)-1] == 'C' {
			stack = stack[:len(stack)-1]
		} else {
			stack = append(stack, byte(ch))
		}
	}
	return len(stack)
}
func removeDuplicates_(s string) string {
	stack := make([]byte, 0, len(s))
	for _, ch := range s {
		if len(stack) > 0 && byte(ch) == stack[len(stack)-1] {
			stack = stack[:len(stack)-1]
		} else {
			stack = append(stack, byte(ch))
		}
	}
	return string(stack)
}
func makeGood(s string) string {
	stack := make([]byte, 0, len(s))
	for _, ch := range s {
		if len(stack) > 0 && (unicode.IsLower(ch) && byte(ch)-stack[len(stack)-1] == 32 || (unicode.IsUpper(ch) && stack[len(stack)-1]-byte(ch) == 32)) {
			stack = stack[:len(stack)-1]
		} else {
			stack = append(stack, byte(ch))
		}
	}
	return string(stack)
}

//func isValid_(s string) bool {
//	for strings.Contains(s, "abc") {
//		s = strings.ReplaceAll(s, "abc", "")
//	}
//	return len(s) == 0
//}

func isValid_(s string) bool {
	stack := make([]byte, 0, len(s))
	for _, ch := range s {
		if ch == 'c' && len(stack) >= 2 && string(stack[len(stack)-2:]) == "bc" {
			stack = stack[:len(stack)-2]
		} else {
			stack = append(stack, byte(ch))
		}
	}
	return len(stack) == 0
}

// 子数组问题中如果出现负数导致失去单调性 可以考虑使用前缀和
func shortestSubarray(nums []int, k int) int {
	prefix := make([]int, len(nums)+1)
	for i := 0; i < len(nums); i++ {
		prefix[i+1] = prefix[i] + nums[i]
	}
	ans := len(nums) + 1
	queue := make([]int, 0)
	for i := 0; i < len(prefix); i++ {
		//维护单调队列：从队尾弹出
		for len(queue) > 0 && prefix[queue[len(queue)-1]] >= prefix[i] {
			queue = queue[:len(queue)-1]
		}
		queue = append(queue, i)

		//从队首弹出 不可能组成更短的子数组
		for len(queue) > 0 && prefix[i]-prefix[queue[0]] >= k {
			ans = min(ans, i-queue[0])
			queue = queue[1:]
		}
	}
	if ans == len(nums)+1 {
		return -1
	}
	return ans
}
func maxOperations(nums []int) int {
	point := nums[0] + nums[1]
	op := 1
	nums = nums[2:]
	for ; len(nums) >= 2; op++ {
		if point == nums[0]+nums[1] {
			nums = nums[2:]
		} else {
			break
		}
	}
	return op
}

func minDeletion(nums []int) (ans int) {
	stack := make([]int, 0, len(nums))
	for _, num := range nums {
		if len(stack)%2 == 1 && stack[len(stack)-1] == num {
			ans++
		} else {
			stack = append(stack, num)
		}
	}
	if (len(nums)-ans)%2 == 1 {
		return ans + 1
	}
	return ans
}

func removeDuplicates__(s string, k int) string {
	type Pair struct {
		ch  byte
		cnt int
	}
	stack := make([]Pair, 0, len(s))
	for _, ch := range s {
		cnt := 1
		if len(stack) > 0 && byte(ch) == stack[len(stack)-1].ch {
			cnt += stack[len(stack)-1].cnt
		}
		stack = append(stack, Pair{
			ch:  byte(ch),
			cnt: cnt,
		})
		if stack[len(stack)-1].cnt >= k {
			stack = stack[:len(stack)-k]
		}
	}
	bytes := make([]byte, 0, len(s))
	for i := 0; i < len(stack); i++ {
		bytes = append(bytes, stack[i].ch)
	}
	return string(bytes)
}

// 只有R+L、S+L、R+S会发生碰撞
// 如果为L且栈为空 跳过
// 如果为L且栈不为空 碰撞 两辆车都静止 如果之前存在R 则也会相撞
// 如果为R/S 入栈
func countCollisions(directions string) (ans int) {
	stack := make([]byte, 0, len(directions))
	for _, direction := range directions {
		if direction == 'S' {
			for len(stack) > 0 && stack[len(stack)-1] == 'R' {
				stack = stack[:len(stack)-1]
				ans += 1
			}
			stack = []byte{'S'}
		} else if direction == 'R' {
			stack = append(stack, 'R')
		} else if len(stack) > 0 {
			if stack[len(stack)-1] == 'R' {
				ans += 2
			} else {
				ans += 1
			}
			stack = stack[:len(stack)-1]
			for len(stack) > 0 && stack[len(stack)-1] == 'R' {
				stack = stack[:len(stack)-1]
				ans += 1
			}
			stack = []byte{'S'}
		}
	}
	return
}

// 左括号直接入栈 右括号弹出匹配
func isValid__(s string) bool {
	stack := make([]byte, 0, len(s))
	for _, ch := range s {
		if ch == '(' || ch == '[' || ch == '{' {
			stack = append(stack, byte(ch))
		} else {
			if len(stack) == 0 || ch == ')' && stack[len(stack)-1] != '(' ||
				(ch == ']' && stack[len(stack)-1] != '[') ||
				(ch == '}' && stack[len(stack)-1] != '{') {
				return false
			} else {
				stack = stack[:len(stack)-1]
			}
		}
	}
	return len(stack) == 0
}
func minAddToMakeValid(s string) int {
	stack := make([]byte, 0, len(s))
	for _, ch := range s {
		if ch == '(' {
			stack = append(stack, byte(ch))
		} else {
			if len(stack) > 0 && stack[len(stack)-1] == '(' {
				stack = stack[:len(stack)-1]
			} else {
				stack = append(stack, byte(ch))
			}
		}
	}
	return len(stack)
}

// 如何分解出每一个元语
// 分解出原语后取出最外层的括号即可
func removeOuterParentheses(s string) string {
	stack := make([]byte, 0, len(s))
	ans := make([]byte, 0, len(s))
	start := 0
	for i, ch := range s {
		if ch == '(' {
			stack = append(stack, '(')
		} else {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			ans = append(ans, s[start+1:i]...)
			start = i + 1
		}
	}
	return string(ans)
}

func maxDepth_(s string) (ans int) {
	stack := make([]byte, 0, len(s))
	for _, ch := range s {
		if ch == '(' {
			stack = append(stack, '(')
			ans = max(ans, len(stack))
		} else if ch == ')' {
			stack = stack[:len(stack)-1]
		}
	}
	return
}
func reverseParentheses(s string) string {
	prevBracket := make([]int, 0, len(s))
	stack := make([]byte, 0, len(s))
	for _, ch := range s {
		if ch == '(' {
			prevBracket = append(prevBracket, len(stack))
			stack = append(stack, '(')
		} else if ch == ')' {
			prevBracketLoc := prevBracket[len(prevBracket)-1]
			prevBracket = prevBracket[:len(prevBracket)-1]
			slices.Reverse(stack[prevBracketLoc+1:])
			stack = append(stack[:prevBracketLoc], stack[prevBracketLoc+1:]...)
		} else {
			stack = append(stack, byte(ch))
		}
	}
	return string(stack)
}
func scoreOfParentheses(s string) (ans int) {
	prevBracket := make([]int, 0, len(s))
	stack := make([]string, 0, len(s))
	for _, ch := range s {
		if ch == '(' {
			prevBracket = append(prevBracket, len(stack))
			stack = append(stack, "(")
		} else {
			prevBracketLoc := prevBracket[len(prevBracket)-1]
			prevBracket = prevBracket[:len(prevBracket)-1]
			if len(stack)-prevBracketLoc == 1 {
				stack = append(stack[:len(stack)-1], "1")
			} else {
				for i := prevBracketLoc + 1; i < len(stack); i++ {
					val, _ := strconv.Atoi(stack[i])
					stack[i] = strconv.Itoa(val * 2)
				}
				stack = append(stack[:prevBracketLoc], stack[prevBracketLoc+1:]...)
			}
		}
	}
	for i := 0; i < len(stack); i++ {
		val, _ := strconv.Atoi(stack[i])
		ans += val
	}
	return
}
func minRemoveToMakeValid(s string) string {
	stack := make([]byte, 0, len(s))
	prevBracket := make([]byte, 0, len(s))
	for _, ch := range s {
		if ch == '(' {
			stack = append(stack, '(')
			prevBracket = append(prevBracket, '(')
		} else if ch == ')' {
			if len(prevBracket) != 0 && prevBracket[len(prevBracket)-1] == '(' {
				stack = append(stack, ')')
				prevBracket = prevBracket[:len(prevBracket)-1]
			}
		} else {
			stack = append(stack, byte(ch))
		}
	}
	if len(prevBracket) != 0 {
		slices.Reverse(stack)
		replace := bytes.Replace(stack, []byte{'('}, []byte{}, len(prevBracket))
		slices.Reverse(replace)
		return string(replace)
	}
	return string(stack)
}
func minSwaps_(s string) int {
	stack := make([]byte, 0, len(s))
	for _, ch := range s {
		if ch == '[' {
			stack = append(stack, '[')
		} else {
			if len(stack) > 0 && stack[len(stack)-1] == '[' {
				stack = stack[:len(stack)-1]
			} else {
				stack = append(stack, ']')
			}
		}
	}
	return (len(stack) + 3) / 4
}

func longestValidParentheses(s string) int {
	stack := make([]int, 0, len(s))
	for i, ch := range s {
		if ch == '(' {
			stack = append(stack, i)
		} else {
			if len(stack) > 0 && s[stack[len(stack)-1]] == '(' {
				stack = stack[:len(stack)-1]
			} else {
				stack = append(stack, i)
			}
		}
	}
	sequence := make([]int, len(s))
	for i := 0; i < len(stack); i++ {
		sequence[stack[i]] = 1
	}
	//求最长的0序列
	ans := 0
	length := 0
	for i := 0; i < len(sequence); i++ {
		if sequence[i] == 0 {
			length++
			ans = max(ans, length)
		} else {
			length = 0
		}
	}
	return ans
}

// 单调栈梦开始的地方
func dailyTemperatures(temperatures []int) []int {
	stack := make([]int, 0, len(temperatures))
	ans := make([]int, len(temperatures))
	for i, temperature := range temperatures {
		for len(stack) > 0 && temperatures[stack[len(stack)-1]] < temperature {
			ans[stack[len(stack)-1]] = i - stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	return ans
}
