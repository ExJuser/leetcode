package main

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
	"unicode"
)

//for debug

func totalFruit(fruits []int) int {
	mp := make(map[int]int)
	var left, ans int
	for right := 0; right < len(fruits); right++ {
		mp[fruits[right]]++
		for ; len(mp) > 2; left++ {
			mp[fruits[left]]--
			if mp[fruits[left]] == 0 {
				delete(mp, fruits[left])
			}
		}
		ans = max(ans, right-left+1)
	}
	return ans
}

func minWindow(s string, t string) string {
	var left int
	var check func(mp map[byte]int) bool
	check = func(mp map[byte]int) bool {
		for _, v := range mp {
			if v > 0 {
				return false
			}
		}
		return true
	}
	mp := make(map[byte]int)
	length := len(s) + 1
	ans := ""
	for _, ch := range t {
		mp[byte(ch)]++
	}
	for right := 0; right < len(s); right++ {
		mp[s[right]]--
		for ; check(mp); left++ {
			if right-left+1 < length {
				length = right - left + 1
				ans = s[left : right+1]
			}
			mp[s[left]]++
		}
	}
	return ans
}

func spiralOrder(matrix [][]int) (ans []int) {
	height, width := len(matrix), len(matrix[0])
	top, bottom, left, right := 0, height-1, 0, width-1
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
	//只剩一行
	if top == bottom {
		for i := left; i <= right; i++ {
			ans = append(ans, matrix[top][i])
		}
	} else if left == right {
		for i := top; i <= bottom; i++ {
			ans = append(ans, matrix[i][left])
		}
	}
	return
}

// 5. 最长回文子串 动态规划
func longestPalindrome(s string) string {
	dp := make([][]int, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(s))
	}
	//dp[i][j]可以由dp[i+1][j-1]推出
	//初始对角线：单个字符的字符串都是回文串
	for i := 0; i < len(s); i++ {
		dp[i][i] = 1
	}
	length := 1
	ans := s[0:1]
	//注意遍历的顺序问题
	for j := 1; j < len(s); j++ {
		for i := 0; i < j; i++ {
			if s[i] == s[j] && dp[i+1][j-1] == j-i-1 {
				dp[i][j] = dp[i+1][j-1] + 2
			} else {
				dp[i][j] = 0
			}
			if dp[i][j] > length {
				length = dp[i][j]
				ans = s[i : j+1]
			}
		}
	}
	return ans
}

// 93. 复原 IP 地址 回溯插入三个点/找到三个切割位置
func restoreIpAddresses(s string) (ans []string) {
	//如何判断是否合法：切割出来的数字处于0-255 不能有前导0
	var dfs func(pre, cur int, path []string)
	var check func(str string) bool
	check = func(str string) bool {
		//如果首位是0 那他必须是零本身
		if str[0] == '0' {
			return len(str) == 1
		}
		val, _ := strconv.Atoi(str)
		return val >= 0 && val <= 255
	}
	dfs = func(pre, cur int, path []string) {
		if cur == len(s) {
			if len(path) == 4 && cur == pre {
				ans = append(ans, strings.Join(path, "."))
			}
			return
		}
		//切割
		//判断是否合法
		if check(s[pre : cur+1]) {
			path = append(path, s[pre:cur+1])
			dfs(cur+1, cur+1, path)
			path = path[:len(path)-1]
		}
		//不切割
		dfs(pre, cur+1, path)
	}
	dfs(0, 0, []string{})
	return
}

func addStrings(num1 string, num2 string) string {
	carry := 0
	bytes1 := []byte(num1)
	bytes2 := []byte(num2)
	res := make([]byte, max(len(num1), len(num2)))
	index := len(res) - 1
	i, j := len(bytes1)-1, len(bytes2)-1
	for ; i >= 0 && j >= 0; i, j = i-1, j-1 {
		val := carry + int(bytes1[i]-'0') + int(bytes2[j]-'0')
		carry = val / 10
		val %= 10
		res[index] = byte(val) + '0'
		index--
	}
	for ; i >= 0; i-- {
		val := carry + int(bytes1[i]-'0')
		carry = val / 10
		val %= 10
		res[index] = byte(val) + '0'
		index--
	}
	for ; j >= 0; j-- {
		val := carry + int(bytes2[j]-'0')
		carry = val / 10
		val %= 10
		res[index] = byte(val) + '0'
		index--
	}
	if carry != 0 {
		return fmt.Sprintf("%d%s", carry, string(res))
	}
	return string(res)
}
func shortestSeq(big []int, small []int) []int {
	//在包含全部元素的情况下移动左窗口
	//如何判断包含了全部元素？
	mp := make(map[int]int)
	for _, num := range small {
		mp[num] = 0
	}
	var left, satisfy int
	var ans []int
	length := len(big) + 1
	for right := 0; right < len(big); right++ {
		if _, ok := mp[big[right]]; ok {
			mp[big[right]]++
			if mp[big[right]] == 1 {
				satisfy++
			}
		}
		for ; satisfy == len(small); left++ {
			if right-left+1 < length {
				length = right - left + 1
				ans = []int{left, right}
			}
			if _, ok := mp[big[left]]; ok {
				mp[big[left]]--
				if mp[big[left]] == 0 {
					satisfy--
				}
			}
		}
	}
	return ans
}

// TimeMap 981. 基于时间的键值存储
type TimeMap struct {
	times []struct {
		timestamp int
		key       string
		value     string
	}
}

func Constructor() TimeMap {
	return TimeMap{
		times: make([]struct {
			timestamp int
			key       string
			value     string
		}, 0),
	}
}

func (this *TimeMap) Set(key string, value string, timestamp int) {
	//set中的timestamp都是递增的
	this.times = append(this.times, struct {
		timestamp int
		key       string
		value     string
	}{timestamp: timestamp, key: key, value: value})
}

func (this *TimeMap) Get(key string, timestamp int) string {
	left, right := 0, len(this.times)-1
	target := len(this.times)
	for left <= right {
		mid := (right-left)/2 + left
		if this.times[mid].timestamp > timestamp {
			right = mid - 1
			target = mid
		} else {
			left = mid + 1
		}
	}
	for i := target - 1; i >= 0; i-- {
		if this.times[i].key == key {
			return this.times[i].value
		}
	}
	return ""
}

func SlicesSum[T Number](nums []T) (sum T) {
	for _, num := range nums {
		sum += num
	}
	return
}

type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~float32 | ~float64
}

func smallestDivisor(nums []int, threshold int) int {
	var check func(divisor int) bool
	check = func(divisor int) bool {
		var ans int
		for i := 0; i < len(nums); i++ {
			ans += (nums[i] + divisor - 1) / divisor
		}
		return ans <= threshold
	}
	//越往右 满足条件的可能性越大
	left, right := max(SlicesSum(nums)/threshold, 1), slices.Max(nums)
	ans := right
	for left <= right {
		mid := (right-left)/2 + left
		if check(mid) {
			ans = mid
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return ans
}
func countGoodStrings(low int, high int, zero int, one int) int {
	var mod int = 1e9 + 7
	dp := make([]int, high+1)
	dp[zero] += 1
	dp[one] += 1
	for i := min(one, zero); i <= high; i++ {
		if i >= zero {
			dp[i] = (dp[i] + dp[i-zero]) % mod
		}
		if i >= one {
			dp[i] = (dp[i] + dp[i-one]) % mod
		}
	}
	var ans int
	for i := low; i <= high; i++ {
		ans = (ans + dp[i]) % mod
	}
	return ans
}
func countHousePlacements(n int) int {
	//放或者不放
	dp := make([][2]int, n)
	dp[0][0] = 1
	dp[0][1] = 1
	for i := 1; i < n; i++ {
		dp[i][0] = dp[i-1][1]
		dp[i][1] = dp[i-1][0] + dp[i-1][1]
	}
	return 0
}
func maximumCostSubstring(s string, chars string, vals []int) int {
	//对于一个字符ch 如果它在chars中 它的价值就是chars中的对应
	valsMap := make(map[byte]int)
	for i, ch := range chars {
		valsMap[byte(ch)] = vals[i]
	}
	dp := make([]int, len(s))
	val, ok := valsMap[s[0]]
	if !ok {
		val = int(s[0] - 'a' + 1)
	}
	dp[0] = max(val, 0)
	ans := dp[0]
	for i := 1; i < len(dp); i++ {
		val, ok := valsMap[s[i]]
		if !ok {
			val = int(s[i] - 'a' + 1)
		}
		dp[i] = max(dp[i-1]+val, val, 0)
		ans = max(ans, dp[i])
	}
	return ans
}

func exist(board [][]byte, word string) bool {
	var used [][]bool
	var dfs func(x, y, index int) bool
	dfs = func(x, y, index int) bool {
		if x < 0 || x >= len(board) || y < 0 || y >= len(board[0]) {
			return false
		}
		if !used[x][y] && board[x][y] == word[index] {
			used[x][y] = true
			if index == len(word)-1 {
				return true
			}
			if dfs(x+1, y, index+1) || dfs(x, y+1, index+1) || dfs(x-1, y, index+1) || dfs(x, y-1, index+1) {
				return true
			}
			used[x][y] = false
		}
		return false
	}
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {
			if board[i][j] == word[0] {
				used = make([][]bool, len(board))
				for i := 0; i < len(used); i++ {
					used[i] = make([]bool, len(board[i]))
				}
				if dfs(i, j, 0) {
					return true
				}
			}
		}
	}
	return false
}
func partition(s string) (ans [][]string) {
	var check func(s string) bool
	//左闭右开区间
	var dfs func(left, right int, path []string)
	check = func(s string) bool {
		bytes := []byte(s)
		slices.Reverse(bytes)
		return string(bytes) == s
	}
	dfs = func(left, right int, path []string) {
		if right == len(s) { //到达字符串末尾
			if check(s[left:right]) {
				path = append(path, s[left:right])
				ans = append(ans, append([]string{}, path...))
			}
			return
		}
		//如果当前切割结果是回文串
		if check(s[left:right]) {
			path = append(path, s[left:right])
			dfs(right, right+1, path)
			path = path[:len(path)-1]
		}
		dfs(left, right+1, path)
	}
	dfs(0, 1, []string{})
	return
}
func decodeString(s string) string {
	numTemp := make([]byte, 0)
	numStack := make([]int, 0)
	temp := make([]byte, 0)
	for _, ch := range s {
		if unicode.IsDigit(ch) {
			numTemp = append(numTemp, byte(ch))
		} else if unicode.IsLetter(ch) {
			temp = append(temp, byte(ch))
		} else if ch == '[' {
			temp = append(temp, byte(ch))
			num, _ := strconv.Atoi(string(numTemp))
			numStack = append(numStack, num)
			numTemp = []byte{}
		} else {
			j := len(temp) - 1
			for temp[j] != '[' {
				j--
			}
			toAppend := string(temp[j+1:])
			temp = temp[:j]
			num := numStack[len(numStack)-1]
			numStack = numStack[:len(numStack)-1]
			for ; num > 0; num-- {
				temp = append(temp, toAppend...)
			}
		}
	}
	return string(temp)
}

// 210. 课程表 II 打印拓扑排序序列
func findOrder(numCourses int, prerequisites [][]int) []int {
	path := make([]int, 0, numCourses)
	graph := make([][]int, numCourses)
	inDegree := make(map[int]int)
	for _, pre := range prerequisites {
		preCourse, course := pre[1], pre[0]
		graph[preCourse] = append(graph[preCourse], course)
		inDegree[course]++
	}

	queue := make([]int, 0, numCourses)
	for i := 0; i < numCourses; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	for len(queue) > 0 {
		temp := queue[0]
		queue = queue[1:]
		path = append(path, temp)
		for _, c := range graph[temp] {
			inDegree[c]--
			if inDegree[c] == 0 {
				queue = append(queue, c)
			}
		}
	}
	if len(path) == numCourses {
		return path
	}
	return []int{}
}

// LCR 114. 火星词典 拓扑排序
func alienOrder(words []string) string {
	graph := make([]map[int]struct{}, 26)
	for i := 0; i < len(graph); i++ {
		graph[i] = make(map[int]struct{})
	}
	inDegree := make(map[int]int)
	set := make(map[int]struct{})
	for _, word := range words {
		for _, ch := range word {
			set[int(ch-'a')] = struct{}{}
		}
	}
	var flag bool
	for i := 0; i < len(words); i++ {
		for j := i + 1; j < len(words); j++ {
			word1 := words[i]
			word2 := words[j]
			//找到word1和word2第一个不相同的字符ch1 ch2
			//加入边ch1->ch2
			k := 0
			for ; k < len(word1) && k < len(word2); k++ {
				if word1[k] != word2[k] {
					if _, ok := graph[word1[k]-'a'][int(word2[k]-'a')]; !ok {
						graph[word1[k]-'a'][int(word2[k]-'a')] = struct{}{}
						inDegree[int(word2[k]-'a')]++
					}
					break
				}
			}
			if (k == len(word1) || k == len(word2)) && len(word1) > len(word2) {
				flag = true
			}
			//if (k == len(word1) && k < len(word2)) || (k == len(word2) && k < len(word1)) {
			//	flag = true
			//}
		}
	}
	ans := make([]byte, 0, 26)
	queue := make([]int, 0, 26)

	for k := range set {
		if inDegree[k] == 0 {
			queue = append(queue, k)
		}
	}

	if flag && len(inDegree) == 0 {
		return ""
	}

	for len(queue) > 0 {
		temp := queue[0]
		queue = queue[1:]
		ans = append(ans, byte(temp+'a'))
		delete(set, temp)
		for e := range graph[temp] {
			inDegree[e]--
			if inDegree[e] == 0 {
				delete(inDegree, e)
				queue = append(queue, e)
			}
		}
	}
	if len(inDegree) != 0 {
		return ""
	}
	for k := range set {
		ans = append(ans, byte(k+'a'))
	}
	return string(ans)
}
func main() {
	fmt.Println(alienOrder([]string{"abc", "ab"}))
}
