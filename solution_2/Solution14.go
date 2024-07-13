package main

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode"
)

//func replaceWords(dictionary []string, sentence string) string {
//	wordDictionary := &WordDictionary{}
//	for _, word := range dictionary {
//		wordDictionary.AddWord(word)
//	}
//	words := strings.Split(sentence, " ")
//	for i, word := range words {
//		p := wordDictionary
//		builder := strings.Builder{}
//		for _, char := range word {
//			index := char - 'a'
//			if p.Children[index] != nil && !p.EndOfWord {
//				builder.WriteByte(byte(char))
//				p = p.Children[index]
//			} else {
//				break
//			}
//		}
//		if builder.Len() > 0 && p.EndOfWord {
//			words[i] = builder.String()
//		}
//	}
//	return strings.Join(words, " ")
//}

//func longestWord(words []string) (ans string) {
//	slices.SortFunc(words, func(a, b string) int {
//		if len(a) == len(b) {
//			return strings.Compare(b, a)
//		}
//		return len(a) - len(b)
//	})
//	dictionary := &WordDictionary{}
//	for _, word := range words {
//		if len(word) == 1 || dictionary.Prefix(word[:len(word)-1]) {
//			dictionary.AddWord(word)
//			ans = word
//		}
//	}
//	return
//}

type Trie struct {
	root *TrieNode
}

//func Constructor() Trie {
//	return Trie{root: &TrieNode{}}
//}
//
//type TrieNode struct {
//	next  [26]*TrieNode
//	isEnd bool
//}
//
//func (t *Trie) Insert(word string) {
//	p := t.root
//	for _, char := range word {
//		index := char - 'a'
//		if p.next[index] == nil {
//			p.next[index] = &TrieNode{}
//		}
//		p = p.next[index]
//	}
//	p.isEnd = true
//}
//
//func (t *Trie) Search(word string) bool {
//	p := t.root
//	for _, char := range word {
//		index := char - 'a'
//		if p.next[index] == nil {
//			return false
//		}
//		p = p.next[index]
//	}
//	return p.isEnd
//}
//
//func (t *Trie) StartsWith(word string) bool {
//	p := t.root
//	for _, char := range word {
//		index := char - 'a'
//		if p.next[index] == nil {
//			return false
//		}
//		p = p.next[index]
//	}
//	return true
//}

// 第一道回溯题
//func combine(n int, k int) (ans [][]int) {
//	var dfs func(start int, path []int)
//	dfs = func(start int, path []int) {
//		if len(path) == k {
//			//避免Go中切片的引用属性导致的bug
//			ans = append(ans, append([]int(nil), path...))
//			return
//		}
//		//剪枝操作：剩余的长度不够
//		for i := start; i <= len(path)+n-k+1; i++ {
//			path = append(path, i)
//			dfs(i+1, path)
//			//回溯操作 回退路径 恢复现场
//			path = path[:len(path)-1]
//		}
//	}
//	dfs(1, []int{})
//	return
//}

//func combine(n int, k int) (ans [][]int) {
//	//n代表数字的范围 k代表多少个数组合
//	var dfs func(start int, path []int)
//	dfs = func(start int, path []int) {
//		if len(path) == k {
//			ans = append(ans, append([]int{}, path...))
//			return
//		}
//		if start > n || start > n+len(path)-k+1 {
//			return
//		}
//		//选或者不选
//		path = append(path, start)
//		dfs(start+1, path)
//		path = path[:len(path)-1]
//
//		dfs(start+1, path)
//	}
//	dfs(1, []int{})
//	return
//}

// 第一道手撕的回溯题
// 非子集型问题没有选或不选的解法
//func letterCombinations(digits string) (ans []string) {
//	mapping := []string{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}
//	//start代表到达了digits中的第几位
//	var dfs func(start int, path []byte)
//	dfs = func(start int, path []byte) {
//		if len(path) == len(digits) {
//			ans = append(ans, string(append([]byte(nil), path...)))
//			return
//		}
//		for _, char := range mapping[digits[start]-'0'] {
//			path = append(path, byte(char))
//			dfs(start+1, path)
//			//恢复现场
//			path = path[:len(path)-1]
//		}
//	}
//	if len(digits) != 0 {
//		dfs(0, []byte{})
//	}
//	return
//}

// 第二道手撕的回溯题
//	func subsets(nums []int) (ans [][]int) {
//		//对不同的长度循环回溯
//		var dfs func(start, n int, path []int)
//		dfs = func(start, n int, path []int) {
//			if len(path) == n {
//				ans = append(ans, append([]int(nil), path...))
//				return
//			}
//			//剪枝操作：剩余的长度不够
//			for i := start; i <= len(path)+len(nums)-n; i++ {
//				path = append(path, nums[i])
//				dfs(i+1, n, path)
//				//恢复现场
//				path = path[:len(path)-1]
//			}
//		}
//		for i := 0; i <= len(nums); i++ {
//			dfs(0, i, []int{})
//		}
//		return
//	}

// 子集型回溯：选或不选的思路
func subsets(nums []int) (ans [][]int) {
	n := len(nums)
	var dfs func(start int, path []int)
	dfs = func(start int, path []int) {
		if start == n {
			ans = append(ans, append([]int(nil), path...))
			return
		}
		dfs(start+1, path)
		path = append(path, nums[start])
		dfs(start+1, path)
		path = path[:len(path)-1]
	}
	dfs(0, []int{})
	return
}

// 第三道手撕的回溯题
//
//	func combinationSum3(k int, n int) (ans [][]int) {
//		var dfs func(start, sum int, path []int)
//		dfs = func(start, sum int, path []int) {
//			if sum > n || len(path) > k {
//				return
//			}
//			if len(path) == k && sum == n {
//				ans = append(ans, append([]int{}, path...))
//				return
//			}
//			//剪枝操作
//			for i := start; i <= min(9, n-sum, len(path)+n-k+1); i++ {
//				sum += i
//				path = append(path, i)
//				dfs(i+1, sum, path)
//				//恢复现场
//				path = path[:len(path)-1]
//				sum -= i
//			}
//		}
//		dfs(1, 0, []int{})
//		return
//	}

//枚举每一种切割的方式 对每一种方式再循环判断是否回文
//func partition(s string) (ans [][]string) {
//	var dfs func(start, length int, path []string)
//	dfs = func(start, length int, path []string) {
//		if start+length > len(s) {
//			return
//		}
//		if start+length == len(s) {
//			path = append(path, s[start:start+length])
//			flag := true
//			for _, p := range path {
//				if !isPalindrome(p) {
//					flag = false
//				}
//			}
//			if flag {
//				ans = append(ans, append([]string{}, path...))
//			}
//			return
//		}
//		dfs(start, length+1, path)
//		path = append(path, s[start:start+length])
//		dfs(start+length, 1, path)
//		path = path[:len(path)-1]
//	}
//	dfs(0, 1, []string{})
//	return
//}

func partition(s string) (ans [][]string) {
	var dfs func(start, length int, path []string)
	dfs = func(start, length int, path []string) {
		if start+length > len(s) {
			return
		}
		//可以切割到最后时 判断最后一个切割是否为回文
		//若为回文 则加入path并更新答案
		if start+length == len(s) && isPalindrome(s[start:start+length]) {
			path = append(path, s[start:start+length])
			ans = append(ans, append([]string{}, path...))
		}
		//切或者不切
		dfs(start, length+1, path)
		if isPalindrome(s[start : start+length]) {
			path = append(path, s[start:start+length])
			dfs(start+length, 1, path)
			//恢复现场
			path = path[:len(path)-1]
		}
	}
	dfs(0, 1, []string{})
	return
}

// 题目限制 因此不需要去重
func combinationSum(candidates []int, target int) (ans [][]int) {
	var dfs func(start, sum int, path []int)
	dfs = func(start, sum int, path []int) {
		if sum > target || start >= len(candidates) || candidates[start] > target {
			return
		}
		if sum == target {
			ans = append(ans, append([]int{}, path...))
			return
		}
		path = append(path, candidates[start])
		sum += candidates[start]
		//可以重复选取 因此不是dfs(start+1, sum, path)
		dfs(start, sum, path)
		sum -= candidates[start]
		path = path[:len(path)-1]

		dfs(start+1, sum, path)
	}
	dfs(0, 0, []int{})
	return
}

// 去重问题
//func combinationSum2(candidates []int, target int) (ans [][]int) {
//	var dfs func(start, sum int, path []int)
//	dfs = func(start, sum int, path []int) {
//		if sum == target {
//			ans = append(ans, append([]int{}, path...))
//			return
//		}
//		if start >= len(candidates) || sum > target || candidates[start] > target {
//			return
//		}
//		sum += candidates[start]
//		path = append(path, candidates[start])
//		dfs(start+1, sum, path)
//		//去重
//		for start < len(candidates)-1 && candidates[start] == candidates[start+1] {
//			start++
//		}
//		sum -= candidates[start]
//		path = path[:len(path)-1]
//		dfs(start+1, sum, path)
//	}
//	slices.Sort(candidates)
//	dfs(0, 0, []int{})
//	return
//}

func isSubIPValid(ip string) bool {
	//若以0开头 长度必须为1
	//整体长度小于等于3
	//值在0-255之间
	if ip[0] == '0' {
		return len(ip) == 1
	}
	if len(ip) > 3 {
		return false
	}
	val, _ := strconv.Atoi(ip)
	return val >= 0 && val <= 255
}

// 切割字符串类型的题目完全拿捏了
// 相同类型的题目：分割回文串 判断是否回文和判断子ip是否合法类似
func restoreIpAddresses(s string) (ans []string) {
	var dfs func(start, length int, path []string)
	dfs = func(start, length int, path []string) {
		if len(path) > 4 || start > len(s) || length > 3 || start+length > len(s) {
			return
		}
		if start+length == len(s) {
			if isSubIPValid(s[start:start+length]) && len(path) == 3 {
				path = append(path, s[start:start+length])
				ans = append(ans, strings.Join(path, "."))
			}
		}
		if isSubIPValid(s[start : start+length]) {
			path = append(path, s[start:start+length])
			dfs(start+length, 1, path)
			path = path[:len(path)-1]
		}
		dfs(start, length+1, path)
	}
	dfs(0, 1, []string{})
	return
}

// 把握好去重的逻辑：为什么会出现重复
func subsetsWithDup(nums []int) (ans [][]int) {
	var dfs func(start int, path []int)
	dfs = func(start int, path []int) {
		if start == len(nums) {
			ans = append(ans, append([]int{}, path...))
			return
		}
		path = append(path, nums[start])
		dfs(start+1, path)
		path = path[:len(path)-1]
		//如果不选 则跳过所有相同的数字 都不选
		for start < len(nums)-1 && nums[start] == nums[start+1] {
			start++
		}
		dfs(start+1, path)
	}
	slices.Sort(nums)
	dfs(0, []int{})
	return
}
func slices2String(path []int) string {
	builder := strings.Builder{}
	for _, num := range path {
		builder.WriteByte(byte(num + '0'))
	}
	return builder.String()
}

// 还是去重问题
func findSubsequences(nums []int) (ans [][]int) {
	set := make(map[string]struct{})
	var dfs func(start int, path []int)
	dfs = func(start int, path []int) {
		if start == len(nums) {
			if len(path) >= 2 {
				if _, ok := set[slices2String(path)]; !ok {
					ans = append(ans, append([]int{}, path...))
					set[slices2String(path)] = struct{}{}
				}
			} else {
				path = []int{}
			}
			return
		}
		if len(path) == 0 || nums[start] >= path[len(path)-1] {
			path = append(path, nums[start])
			dfs(start+1, path)
			path = path[:len(path)-1]
		}
		for start < len(nums)-1 && nums[start] == nums[start+1] {
			start++
		}
		dfs(start+1, path)
	}
	dfs(0, []int{})
	return
}

func isParenthesesValid(pattern string) bool {
	stack := make([]byte, 0, len(pattern))
	for _, char := range pattern {
		if char == '(' || len(stack) == 0 {
			stack = append(stack, byte(char))
		} else {
			if stack[len(stack)-1] == '(' {
				stack = stack[:len(stack)-1]
			} else {
				return false
			}
		}
	}
	return len(stack) == 0
}

//	func generateParenthesis(n int) (ans []string) {
//		var dfs func(count int, path []byte)
//		dfs = func(count int, path []byte) {
//			if count == 2*n && isParenthesesValid(string(path)) {
//				ans = append(ans, string(append([]byte{}, path...)))
//				return
//			}
//			if count > 2*n {
//				return
//			}
//			//若左括号
//			path = append(path, '(')
//			dfs(count+1, path)
//			path = path[:len(path)-1]
//
//			//若右括号
//			path = append(path, ')')
//			dfs(count+1, path)
//			path = path[:len(path)-1]
//		}
//		dfs(0, []byte{})
//		return
//	}
func generateParenthesis(n int) (ans []string) {
	var dfs func(left, right int, path []byte)
	dfs = func(left, right int, path []byte) {
		if left > n || right > n {
			return
		}
		if left+right == 2*n && isParenthesesValid(string(path)) {
			ans = append(ans, string(append([]byte{}, path...)))
			return
		}
		if left < n {
			path = append(path, '(')
			dfs(left+1, right, path)
			path = path[:len(path)-1]
		}
		if right < n {
			path = append(path, ')')
			dfs(left, right+1, path)
			path = path[:len(path)-1]
		}
	}
	dfs(0, 0, []byte{})
	return
}

// 用map的set记录可以使用的值
// 也可以用一个数组标记是否可以使用
func permute(nums []int) (ans [][]int) {
	var dfs func(path []int, set map[int]struct{})
	dfs = func(path []int, set map[int]struct{}) {
		if len(path) == len(nums) {
			ans = append(ans, append([]int{}, path...))
			return
		}
		for k, _ := range set {
			path = append(path, k)
			setCopy := make(map[int]struct{})
			for k_, _ := range set {
				setCopy[k_] = struct{}{}
			}
			delete(setCopy, k)
			dfs(path, setCopy)
			path = path[:len(path)-1]
		}
	}
	set := make(map[int]struct{})
	for _, num := range nums {
		set[num] = struct{}{}
	}
	dfs([]int{}, set)
	return
}

// 用set去重太慢了
func permuteUnique(nums []int) (ans [][]int) {
	set := make(map[string]struct{})
	var dfs func(path []int, isUsed []bool)
	dfs = func(path []int, isUsed []bool) {
		if len(path) == len(nums) {
			if _, ok := set[slices2String(path)]; !ok {
				ans = append(ans, append([]int{}, path...))
				set[slices2String(path)] = struct{}{}
				return
			}
		}
		//从可选的数字中选择一个
		for i := 0; i < len(isUsed); i++ {
			//找到的没用过的这个数不能和前面的重复
			if i > 0 && isUsed[i-1] && nums[i] == nums[i-1] {
				continue
			}
			if !isUsed[i] {
				path = append(path, nums[i])
				isUsed[i] = true
				dfs(path, append([]bool{}, isUsed...))
				path = path[:len(path)-1]
				isUsed[i] = false
			}
		}
	}
	slices.Sort(nums)
	isUsed := make([]bool, len(nums))
	dfs([]int{}, isUsed)
	return
}

func checkDiag(x, y int, chessboard [][]string) bool {
	for i := 0; i < len(chessboard); i++ {
		for j := 0; j < len(chessboard[i]); j++ {
			if (i+j == x+y || i-j == x-y) && chessboard[i][j] == "Q" {
				return false
			}
		}
	}
	return true
}

func checkRow(x int, chessboard [][]string) bool {
	for i := 0; i < len(chessboard); i++ {
		if chessboard[x][i] == "Q" {
			return false
		}
	}
	return true
}

func checkCol(y int, chessboard [][]string) bool {
	for i := 0; i < len(chessboard); i++ {
		if chessboard[i][y] == "Q" {
			return false
		}
	}
	return true
}
func initChessboard(n int) [][]string {
	chessboard := make([][]string, 0, n)
	for i := 0; i < n; i++ {
		chessboard = append(chessboard, make([]string, n))
	}
	for i := 0; i < len(chessboard); i++ {
		for j := 0; j < len(chessboard[0]); j++ {
			chessboard[i][j] = "."
		}
	}
	return chessboard
}

// N皇后
// 每次在一行中寻找可以摆放皇后的位置
// 然后直接进入下一行
func solveNQueens(n int) (ans [][]string) {
	var dfs func(chessboard [][]string, r, placedQueens int)
	dfs = func(chessboard [][]string, r, placedQueens int) {
		if placedQueens == n {
			temp := make([]string, 0, n)
			for _, row := range chessboard {
				temp = append(temp, strings.Join(row, ""))
			}
			ans = append(ans, append([]string{}, temp...))
		}
		for i := 0; i < n; i++ {
			if checkCol(i, chessboard) && checkDiag(r, i, chessboard) {
				chessboard[r][i] = "Q"
				dfs(chessboard, r+1, placedQueens+1)
				chessboard[r][i] = "."
			}
		}
	}
	dfs(initChessboard(n), 0, 0)
	return
}

func initSudoku(board [][]byte) (row, col, subBox [9][9]bool) {
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if unicode.IsDigit(rune(board[i][j])) {
				row[i][board[i][j]-'0'-1] = true
				col[j][board[i][j]-'0'-1] = true
				subBox[i/3*3+j/3][board[i][j]-'0'-1] = true
			}
		}
	}
	return
}

func solveSudoku(board [][]byte) {
	row, col, subBox := initSudoku(board)
	var dfs func(r, c int) bool
	dfs = func(r, c int) bool {
		if r == 9 {
			return true
		}
		if unicode.IsDigit(rune(board[r][c])) {
			if c < 8 {
				return dfs(r, c+1)
			} else {
				return dfs(r+1, 0)
			}
		}
		temp := make([]int, 0, 9)
		for i := 1; i <= 9; i++ {
			if !row[r][i-1] && !col[c][i-1] && !subBox[r/3*3+c/3][i-1] {
				temp = append(temp, i)
			}
		}
		for _, num := range temp {
			row[r][num-1] = true
			col[c][num-1] = true
			subBox[r/3*3+c/3][num-1] = true
			board[r][c] = byte(num + '0')
			if c < 8 && dfs(r, c+1) {
				return true
			} else if c >= 8 && dfs(r+1, 0) {
				return true
			}
			board[r][c] = '.'
			row[r][num-1] = false
			col[c][num-1] = false
			subBox[r/3*3+c/3][num-1] = false
		}
		return false
	}
	dfs(0, 0)
}
func isValidSudoku(board [][]byte) bool {
	row := [9][9]bool{}
	col := [9][9]bool{}
	subBox := [9][9]bool{}
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if board[i][j] != '.' {
				if row[i][int(board[i][j]-'0'-1)] || col[j][int(board[i][j]-'0'-1)] || subBox[i/3*3+j/3][int(board[i][j]-'0'-1)] {
					return false
				}
				row[i][int(board[i][j]-'0'-1)] = true
				col[j][int(board[i][j]-'0'-1)] = true
				subBox[i/3*3+j/3][int(board[i][j]-'0'-1)] = true
			}
		}
	}
	return true
}

/*
*
遇到左括号 pre可以保留
*/
func longestValidParentheses(s string) int {
	invalid := make([]int, len(s))
	stack := make([]int, 0, len(s))
	for i, char := range s {
		if char == '(' {
			stack = append(stack, i)
		} else {
			if len(stack) == 0 {
				invalid[i] = 1
			} else {
				stack = stack[:len(stack)-1]
			}
		}
	}
	for i := 0; i < len(stack); i++ {
		invalid[stack[i]] = 1
	}
	//寻找最长的0
	var maxLen, length int
	for i := 0; i < len(invalid); i++ {
		if invalid[i] == 0 {
			length++
			maxLen = max(maxLen, length)
		} else {
			length = 0
		}
	}
	return maxLen
}

// 还是去重问题
func readBinaryWatch(turnedOn int) (ans []string) {
	set := make(map[int]struct{})
	hour := [4]bool{}
	minute := [6]bool{}
	var dfs func(hCount, mCount, hours, minutes int)
	dfs = func(hCount, mCount, hours, minutes int) {
		if hours > 11 || minutes > 59 || hCount+mCount > turnedOn {
			return
		}
		if hCount+mCount == turnedOn {
			if _, ok := set[hours*100+minutes]; !ok {
				ans = append(ans, fmt.Sprintf("%d:%02d", hours, minutes))
				set[hours*100+minutes] = struct{}{}
			}
			return
		}
		for j := 0; j < len(minute); j++ {
			if !minute[j] {
				minute[j] = true
				dfs(hCount, mCount+1, hours, minutes+1<<j)
				minute[j] = false
			}
		}
		for i := 0; i < len(hour); i++ {
			if !hour[i] {
				hour[i] = true
				dfs(hCount+1, mCount, hours+1<<i, minutes)
				hour[i] = false
			}
		}
	}
	dfs(0, 0, 0, 0)
	return
}

// 回溯求子集
func subsetXORSum(nums []int) int {
	sum := 0
	var dfs func(i int, path []int)
	dfs = func(i int, path []int) {
		if i == len(nums) {
			xor := 0
			for _, sub := range path {
				xor ^= sub
			}
			sum += xor
			return
		}
		path = append(path, nums[i])
		dfs(i+1, path)
		path = path[:len(path)-1]
		dfs(i+1, path)
	}
	dfs(0, []int{})
	return sum
}
func locateStart(board [][]byte, start byte) (ans [][2]int) {
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if board[i][j] == start {
				ans = append(ans, [2]int{i, j})
			}
		}
	}
	return
}
func exist(board [][]byte, word string) bool {
	used := make([][]bool, 0, len(board))
	for i := 0; i < len(board); i++ {
		used = append(used, make([]bool, len(board[0])))
	}
	var dfs func(i, j int, path []byte) bool
	dfs = func(i, j int, path []byte) bool {
		if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) || !strings.HasPrefix(word, string(path)) {
			return false
		}
		if used[i][j] {
			return false
		}
		used[i][j] = true
		path = append(path, board[i][j])
		if word == string(path) {
			return true
		}
		if dfs(i, j+1, path) || dfs(i+1, j, path) || dfs(i-1, j, path) || dfs(i, j-1, path) {
			return true
		} else {
			path = path[:len(path)-1]
			used[i][j] = false
		}
		return false
	}
	start := locateStart(board, word[0])
	for i := 0; i < len(start); i++ {
		if dfs(start[i][0], start[i][1], []byte{}) {
			return true
		}
	}
	return false
}
func letterCasePermutation(s string) (ans []string) {
	bytes := []byte(s)
	var dfs func(i int, path []byte)
	dfs = func(i int, path []byte) {
		if i == len(s) {
			ans = append(ans, string(append([]byte{}, path...)))
			return
		}
		if unicode.IsDigit(rune(bytes[i])) {
			path = append(path, bytes[i])
			dfs(i+1, path)
		} else {
			//每个字母大写或小写
			bytes[i] = byte(unicode.ToLower(rune(bytes[i])))
			path = append(path, bytes[i])
			dfs(i+1, path)
			path = path[:len(path)-1]

			bytes[i] = byte(unicode.ToUpper(rune(bytes[i])))
			path = append(path, bytes[i])
			dfs(i+1, path)
		}
	}
	dfs(0, []byte{})
	return
}
func allPathsSourceTarget(graph [][]int) (ans [][]int) {
	isUsed := make([]bool, len(graph))
	//start代表当前起点
	var dfs func(start int, path []int)
	dfs = func(start int, path []int) {
		if start == len(graph)-1 {
			ans = append(ans, append([]int{}, path...))
			return
		}
		if len(path) > len(graph) {
			return
		}
		//寻找以当前点为起点的路径
		for _, p := range graph[start] {
			if !isUsed[p] {
				isUsed[p] = true
				path = append(path, p)
				dfs(p, path)
				isUsed[p] = false
				path = path[:len(path)-1]
			}
		}
	}
	dfs(0, []int{0})
	return
}
func numsSameConsecDiff(n int, k int) (ans []int) {
	set := make(map[int]struct{})
	var dfs func(i, res int)
	dfs = func(i, res int) {
		if i == n-1 {
			if _, ok := set[res]; !ok {
				ans = append(ans, res)
				set[res] = struct{}{}
			}
			return
		}
		if res%10+k <= 9 {
			dfs(i+1, res*10+res%10+k)
		}
		if res%10-k >= 0 {
			dfs(i+1, res*10+res%10-k)
		}
	}
	for i := 1; i <= 9; i++ {
		dfs(0, i)
	}
	return
}
func findDifferentBinaryString(nums []string) (ans string) {
	length := len(nums)
	var dfs func(i int, path []byte) bool
	dfs = func(i int, path []byte) bool {
		if i == length {
			if !slices.Contains(nums, string(path)) {
				ans = string(path)
				return true
			}
			return false
		}
		path = append(path, '1')
		if dfs(i+1, path) {
			return true
		}
		path = path[:len(path)-1]

		path = append(path, '0')

		if dfs(i+1, path) {
			return true
		}

		path = path[:len(path)-1]
		return false
	}
	dfs(0, []byte{})
	return
}
func minDiffInBST(root *TreeNode) int {
	ans := math.MaxInt
	pre := -1
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		if pre != -1 {
			ans = min(ans, Abs(node.Val-pre))
		}
		pre = node.Val
		fmt.Println(pre)
		dfs(node.Right)
	}
	dfs(root)
	return ans
}
func isValidBST(root *TreeNode) bool {
	var dfs func(node *TreeNode, left, right int) bool
	dfs = func(node *TreeNode, left, right int) bool {
		if node == nil {
			return true
		}
		if node.Val >= right || node.Val <= left {
			return false
		}
		return dfs(node.Left, left, node.Val) && dfs(node.Right, node.Val, right)
	}
	return dfs(root, math.MinInt, math.MaxInt)
}
func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil || q == nil {
		return p == q
	}
	return p.Val == q.Val && isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
}
