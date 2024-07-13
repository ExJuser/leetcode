package main

import (
	"container/heap"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

func trap(height []int) int {
	stack := make([]int, 0, len(height))
	ans := 0
	for i, h := range height {
		if len(stack) == 0 || h <= height[stack[len(stack)-1]] {
			stack = append(stack, i)
		} else {
			preHeight := 0
			for len(stack) > 0 && height[stack[len(stack)-1]] <= h {
				preHeight = height[stack[len(stack)-1]]
				stack = stack[:len(stack)-1]
				if len(stack) == 0 {
					break
				}
				ans += (min(h, height[stack[len(stack)-1]]) - preHeight) * (i - stack[len(stack)-1] - 1)
			}
			stack = append(stack, i)
		}
	}
	return ans
}
func find132pattern(nums []int) bool {
	m := make([]int, len(nums))
	stack := make([]int, 0, len(nums))
	for i, num := range nums {
		if i > 0 {
			m[i] = min(m[i-1], num)
		} else {
			m[i] = min(math.MaxInt, num)
		}
		for len(stack) > 0 && nums[stack[len(stack)-1]] <= num {
			stack = stack[:len(stack)-1]
		}
		if len(stack) > 0 && num > m[stack[len(stack)-1]] {
			return true
		}
		stack = append(stack, i)
	}
	return false
}
func nextGreaterElements(nums []int) []int {
	stack := make([]int, 0, len(nums))
	ans := make([]int, len(nums))
	for i := 0; i < len(ans); i++ {
		ans[i] = -1
	}
	for i, num := range nums {
		for len(stack) > 0 && nums[stack[len(stack)-1]] < num {
			ans[stack[len(stack)-1]] = num
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	for i, num := range nums {
		for len(stack) > 0 && nums[stack[len(stack)-1]] < num {
			if ans[stack[len(stack)-1]] == -1 {
				ans[stack[len(stack)-1]] = num
			}
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	return ans
}
func maxChunksToSorted(arr []int) int {
	var ans, maxRight int
	for i, num := range arr {
		maxRight = max(maxRight, num)
		if i >= maxRight {
			ans++
		}
	}
	return ans
}

type Car struct {
	Position int
	Time     float64
}

func carFleet(target int, position []int, speed []int) int {
	cars := make([]Car, 0, len(position))
	for i := 0; i < len(position); i++ {
		cars = append(cars, Car{
			Position: position[i],
			Time:     float64(target-position[i]) / float64(speed[i]),
		})
	}
	slices.SortFunc(cars, func(a, b Car) int {
		return b.Position - a.Position
	})
	stack := make([]float64, 0, len(cars))
	for _, car := range cars {
		if len(stack) == 0 || stack[len(stack)-1] < car.Time {
			stack = append(stack, car.Time)
		}
	}
	return len(stack)
}

type Bucket struct {
	bucketID int
	time     int
}
type BucketHeap []Bucket

func (b BucketHeap) Len() int {
	return len(b)
}

func (b BucketHeap) Less(i, j int) bool {
	return b[i].time > b[j].time
}

func (b BucketHeap) Swap(i, j int) {
	b[i], b[j] = b[j], b[i]
}

func (b *BucketHeap) Push(x any) {
	*b = append(*b, x.(Bucket))
}

func (b *BucketHeap) Pop() any {
	x := (*b)[b.Len()-1]
	*b = (*b)[:b.Len()-1]
	return x
}
func storeWater(bucket []int, vat []int) int {
	hp := &BucketHeap{}
	for i := 0; i < len(bucket); i++ {
		time := 0
		if vat[i] == 0 {
			time = 0
		} else if bucket[i] == 0 {
			time = 10000
		} else {
			time = int(math.Ceil(float64(vat[i]) / float64(bucket[i])))
		}
		heap.Push(hp, Bucket{bucketID: i, time: time})
	}
	ans := (*hp)[0].time
	storeTimes := 0
	for storeTimes < ans {
		storeTimes++
		bucket[(*hp)[0].bucketID]++
		(*hp)[0].time = int(math.Ceil(float64(vat[(*hp)[0].bucketID]) / float64(bucket[(*hp)[0].bucketID])))
		heap.Fix(hp, 0)
		ans = min(ans, (*hp)[0].time+storeTimes)
	}
	return ans
}

//func maximumScore(a int, b int, c int) int {
//	hp := &IntHeap{}
//	*hp = []int{a, b, c}
//	heap.Init(hp)
//	score := 0
//	for hp.Len() > 1 {
//		x := heap.Pop(hp)
//		y := heap.Pop(hp)
//		if x.(int)-1 > 0 {
//			heap.Push(hp, x.(int)-1)
//		}
//		if y.(int)-1 > 0 {
//			heap.Push(hp, y.(int)-1)
//		}
//		score++
//	}
//	return score
//}
//
//type SeatManager struct {
//	hp *IntHeap
//}
//
//func Constructor(n int) SeatManager {
//	arr := make([]int, n)
//	for i := 0; i < n; i++ {
//		arr[i] = i + 1
//	}
//	hp := &IntHeap{}
//	*hp = arr
//	heap.Init(hp)
//	return SeatManager{hp: hp}
//}

//func (this *SeatManager) Reserve() int {
//	return heap.Pop(this.hp).(int)
//}
//
//func (this *SeatManager) Unreserve(seatNumber int) {
//	heap.Push(this.hp, seatNumber)
//}

type StringHeap []string

func (s StringHeap) Len() int {
	return len(s)
}

func (s StringHeap) Less(i, j int) bool {
	if len(s[i]) == len(s[j]) {
		return strings.Compare(s[i], s[j]) <= 0
	} else {
		return len(s[i]) < len(s[j])
	}
}

func (s StringHeap) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s *StringHeap) Push(x any) {
	*s = append(*s, x.(string))
}

func (s *StringHeap) Pop() any {
	x := (*s)[s.Len()-1]
	*s = (*s)[:s.Len()-1]
	return x
}
func kthLargestNumber(nums []string, k int) string {
	slices.SortFunc(nums, func(a, b string) int {
		if len(a) == len(b) {
			return strings.Compare(a, b)
		} else {
			return len(a) - len(b)
		}
	})
	return nums[len(nums)-k]
}
func maximumImportance(n int, roads [][]int) (ans int64) {
	count := make([]int, n)
	for _, road := range roads {
		count[road[0]]++
		count[road[1]]++
	}
	slices.Sort(count)
	for i := 0; i < n; i++ {
		ans += int64((i + 1) * count[i])
	}
	return
}

//func maxKelements(nums []int, k int) (ans int64) {
//	hp := &IntHeap{}
//	*hp = nums
//	heap.Init(hp)
//	for k > 0 {
//		top := int64(heap.Pop(hp).(int))
//		ans += top
//		heap.Push(hp, int(math.Ceil(float64(top)/3)))
//		k--
//	}
//	return
//}

type FloatHeap []float64

func (f FloatHeap) Len() int {
	return len(f)
}

func (f FloatHeap) Less(i, j int) bool {
	return f[i] > f[j]
}

func (f FloatHeap) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}

func (f *FloatHeap) Push(x any) {
	*f = append(*f, x.(float64))
}

func (f *FloatHeap) Pop() any {
	x := (*f)[f.Len()-1]
	*f = (*f)[:f.Len()-1]
	return x
}

func halveArray(nums []int) (ans int) {
	hp := &FloatHeap{}
	var sum float64
	for _, num := range nums {
		heap.Push(hp, float64(num))
		sum += float64(num)
	}
	curSum := sum
	for curSum > sum/2 {
		ans++
		(*hp)[0] /= 2
		curSum -= (*hp)[0]
		heap.Fix(hp, 0)
	}
	return
}

type Creator struct {
	name       string
	mostViews  int
	totalViews int
	mostViewed string
}

func mostPopularCreator(creators []string, ids []string, views []int) (ans [][]string) {
	creatorMap := make(map[string]*Creator)
	for i := 0; i < len(creators); i++ {
		if _, ok := creatorMap[creators[i]]; !ok {
			creatorMap[creators[i]] = &Creator{name: creators[i], mostViews: views[i], totalViews: views[i], mostViewed: ids[i]}
		} else {
			creatorMap[creators[i]].totalViews += views[i]
			if views[i] > creatorMap[creators[i]].mostViews {
				creatorMap[creators[i]].mostViews = views[i]
				creatorMap[creators[i]].mostViewed = ids[i]
			} else if views[i] == creatorMap[creators[i]].mostViews {
				if strings.Compare(ids[i], creatorMap[creators[i]].mostViewed) < 0 {
					creatorMap[creators[i]].mostViewed = ids[i]
				}
			}
		}
	}
	res := make([]Creator, 0, len(creatorMap))
	for _, v := range creatorMap {
		res = append(res, *v)
	}
	slices.SortFunc(res, func(a, b Creator) int {
		return b.totalViews - a.totalViews
	})
	index := 0
	for index < len(res) && res[index].totalViews == res[0].totalViews {
		ans = append(ans, []string{res[index].name, res[index].mostViewed})
		index++
	}
	return ans
}

//	type NumberContainers struct {
//		container map[int]*IntHeap
//		location  map[int]int
//	}
//
//	func Constructor() NumberContainers {
//		return NumberContainers{
//			container: make(map[int]*IntHeap),
//			location:  make(map[int]int),
//		}
//	}
//
//	func (this *NumberContainers) Change(index int, number int) {
//		if this.location[index] != 0 {
//			hp := this.container[this.location[index]]
//			for i := 0; i < hp.Len(); i++ {
//				if (*hp)[i] == index {
//					heap.Remove(hp, i)
//					break
//				}
//			}
//		}
//		if _, ok := this.container[number]; !ok {
//			this.container[number] = &IntHeap{}
//		}
//		heap.Push(this.container[number], index)
//		this.location[index] = number
//	}
//
//	func (this *NumberContainers) Find(number int) int {
//		if this.container[number] == nil || this.container[number].Len() == 0 {
//			return -1
//		} else {
//			return (*this.container[number])[0]
//		}
//	}

//	type LUPrefix struct {
//		hp       *IntHeap
//		uploaded []int
//	}
//
//	func Constructor(n int) LUPrefix {
//		arr := make([]int, n)
//		for i := 0; i < n; i++ {
//			arr[i] = i + 1
//		}
//		hp := &IntHeap{}
//		*hp = arr
//		heap.Init(hp)
//		return LUPrefix{hp: hp, uploaded: make([]int, n)}
//	}
//
//	func (this *LUPrefix) Upload(video int) {
//		this.uploaded[video-1] = 1
//	}
//
//	func (this *LUPrefix) Longest() int {
//		for this.hp.Len() > 0 && this.uploaded[(*this.hp)[0]-1] == 1 {
//			heap.Pop(this.hp)
//		}
//		if this.hp.Len() == 0 {
//			return len(this.uploaded)
//		} else {
//			return (*this.hp)[0] - 1
//		}
//	}
func largestPerimeter(nums []int) int {
	slices.SortFunc(nums, func(a, b int) int {
		return b - a
	})
	for i := 0; i <= len(nums)-3; i++ {
		if nums[i] < nums[i+1]+nums[i+2] {
			return nums[i] + nums[i+1] + nums[i+2]
		}
	}
	return 0
}
func digitSum(n int) int {
	sum := 0
	for n > 0 {
		sum += n % 10
		n /= 10
	}
	return sum
}
func countLargestGroup(n int) int {
	digitSumMap := make(map[int][]int)
	maximum := math.MinInt
	for i := 1; i <= n; i++ {
		sum := digitSum(i)
		digitSumMap[sum] = append(digitSumMap[sum], i)
		maximum = max(maximum, len(digitSumMap[sum]))
	}
	ans := 0
	for _, v := range digitSumMap {
		if len(v) == maximum {
			ans++
		}
	}
	return ans
}
func buddyStrings(s string, goal string) bool {
	if len(s) != len(goal) {
		return false
	}
	if s == goal {
		count := make(map[byte]int)
		for _, char := range s {
			count[byte(char)]++
			if count[byte(char)] >= 2 {
				return true
			}
		}
		return false
	} else {
		diffCharS := make([]byte, 0)
		diffCharGoal := make([]byte, 0)
		for i := 0; i < len(s); i++ {
			if s[i] != goal[i] {
				if len(diffCharS) >= 2 {
					return false
				}
				diffCharS = append(diffCharS, s[i])
				diffCharGoal = append(diffCharGoal, goal[i])
			}
		}
		if len(diffCharS) != 2 {
			return false
		} else if diffCharS[0] != diffCharGoal[1] || diffCharS[1] != diffCharGoal[0] {
			return false
		} else {
			return true
		}
	}
}
func numOfPairs(nums []string, target string) (ans int) {
	for i := 0; i < len(nums); i++ {
		if strings.HasPrefix(target, nums[i]) {
			for j := 0; j < len(nums); j++ {
				if i != j && strings.HasSuffix(target, nums[j]) {
					if nums[i]+nums[j] == target {
						ans++
					}
				}
			}
		}
	}
	return
}
func secondHighest(s string) int {
	set := make(map[int]struct{})
	for _, char := range s {
		if unicode.IsNumber(char) {
			set[int(char-'0')] = struct{}{}
		}
	}
	nums := make([]int, 0, len(set))
	for k, _ := range set {
		nums = append(nums, k)
	}
	slices.Sort(nums)
	if len(nums) < 2 {
		return -1
	} else {
		return nums[len(nums)-2]
	}
}

// 暴力
func numTeams(rating []int) int {
	cnt := 0
	for j := 1; j < len(rating)-1; j++ {
		for i := 0; i < j; i++ {
			for k := j + 1; k < len(rating); k++ {
				if (rating[i]-rating[j])*(rating[j]-rating[k]) > 0 {
					cnt++
				}
			}
		}
	}
	return cnt
}

//暴力
//func makeGood(s string) string {
//	for i := 0; i < len(s); {
//		if i+1 < len(s) && s[i] != s[i+1] && unicode.ToLower(rune(s[i])) == unicode.ToLower(rune(s[i+1])) {
//			s = s[:i] + s[i+2:]
//			i = 0
//		} else {
//			i++
//		}
//	}
//	return s
//}

// 栈
func makeGood(s string) string {
	stack := make([]byte, 0, len(s))
	for _, char := range s {
		if len(stack) > 0 && byte(char) != stack[len(stack)-1] && unicode.ToLower(char) == unicode.ToLower(rune(stack[len(stack)-1])) {
			stack = stack[:len(stack)-1]
		} else {
			stack = append(stack, byte(char))
		}
	}
	return string(stack)
}
func maxOperations(nums []int, k int) int {
	count := make(map[int]int)
	for _, num := range nums {
		count[num]++
	}
	ans := 0
	for key, value := range count {
		cnt := min(value, count[k-key])
		if key == k-key {
			cnt /= 2
		}
		count[key] -= cnt
		count[k-key] -= cnt
		ans += cnt
	}
	return ans
}
func minimumMoves(s string) int {
	cnt := 0
	for i := 0; i < len(s); {
		if s[i] == 'X' {
			cnt++
			i += 3
		} else {
			i += 1
		}
	}
	return cnt
}
func largestWordCount(messages []string, senders []string) string {
	maxWord := math.MinInt
	maxSender := ""
	hash := make(map[string]int)
	for i, message := range messages {
		wordCount := len(strings.Split(message, " "))
		hash[senders[i]] += wordCount
		if hash[senders[i]] > maxWord {
			maxSender = senders[i]
			maxWord = hash[senders[i]]
		} else if hash[senders[i]] == maxWord {
			if strings.Compare(senders[i], maxSender) >= 0 {
				maxSender = senders[i]
			}
		}
	}
	return maxSender
}
func removeStars(s string) string {
	stack := make([]byte, 0, len(s))
	for _, char := range s {
		if char == '*' {
			stack = stack[:len(stack)-1]
		} else {
			stack = append(stack, byte(char))
		}
	}
	return string(stack)
}

//	func twoCitySchedCost(costs [][]int) int {
//		slices.SortFunc(costs, func(a, b []int) int {
//			return (a[0] - a[1]) - (b[0] - b[1])
//		})
//		ans := 0
//		for i := 0; i < len(costs); i++ {
//			if i < len(costs)/2 {
//				ans += costs[i][0]
//			} else {
//				ans += costs[i][1]
//			}
//		}
//		return ans
//	}
func splitNum(num int) int {
	num1 := 0
	num2 := 0
	k := 1
	numsArr := make([]int, 0)
	for num > 0 {
		numsArr = append(numsArr, num%10)
		num /= 10
	}
	slices.Sort(numsArr)
	for i := len(numsArr) - 1; i >= 1; i -= 2 {
		num1 += k * numsArr[i]
		num2 += k * numsArr[i-1]
		k *= 10
	}
	if len(numsArr)%2 != 0 {
		num1 += k * numsArr[0]
	}
	return num1 + num2
}
func kthDistinct(arr []string, k int) string {
	count := make(map[string]int)
	for _, str := range arr {
		count[str]++
	}
	for i := 0; i < len(arr); i++ {
		if count[arr[i]] == 1 {
			if k == 1 {
				return arr[i]
			} else {
				k--
			}
		}
	}
	return ""
}
func numSub(s string) int {
	ans := 0
	ones := strings.Split(s, "0")
	for _, one := range ones {
		ans += len(one) * (len(one) + 1) / 2
	}
	return ans % (1000000007)
}

//暴力
//func countQuadruplets(nums []int) int {
//	cnt := 0
//	for a := 0; a < len(nums); a++ {
//		for b := a + 1; b < len(nums); b++ {
//			for c := b + 1; c < len(nums); c++ {
//				for d := c + 1; d < len(nums); d++ {
//					if nums[a]+nums[b]+nums[c] == nums[d] {
//						cnt++
//					}
//				}
//			}
//		}
//	}
//	return cnt
//}

// 以为能优化 实际上更慢了
func countQuadruplets(nums []int) int {
	hash := make(map[int][]int)
	for i, num := range nums {
		hash[num] = append(hash[num], i)
	}
	cnt := 0
	for a := 0; a < len(nums); a++ {
		for b := a + 1; b < len(nums); b++ {
			for c := b + 1; c < len(nums); c++ {
				target := nums[a] + nums[b] + nums[c]
				cnt += len(hash[target]) - sort.SearchInts(hash[target], c+1)
			}
		}
	}
	return cnt
}

func shiftingLetters(s string, shifts []int) string {
	suffixSum := make([]int, len(shifts))
	sum := 0
	for i := len(shifts) - 1; i >= 0; i-- {
		sum += shifts[i]
		suffixSum[i] = sum
	}
	bytes := []byte(s)
	for i := 0; i < len(bytes); i++ {
		bytes[i] += byte(suffixSum[i] % 26)
		if bytes[i] > 'z' {
			bytes[i] = bytes[i] - 'z' + 'a' - 1
		}
	}
	return string(bytes)
}
func waysToBuyPensPencils(total int, cost1 int, cost2 int) int64 {
	var ans int64
	cost1, cost2 = max(cost1, cost2), min(cost1, cost2)
	for pen := 0; pen <= total/cost1; pen++ {
		ans += int64(total-pen*cost1)/int64(cost2) + 1
	}
	return ans
}

//	func gcdOfStrings(str1 string, str2 string) string {
//		length := gcd(len(str1), len(str2))
//		if str1[:length] != str2[:length] {
//			return ""
//		} else {
//			builder1 := strings.Builder{}
//			builder2 := strings.Builder{}
//			gcdString := str1[:length]
//			for i := 0; i < len(str1)/length; i++ {
//				builder1.WriteString(gcdString)
//			}
//			for i := 0; i < len(str2)/length; i++ {
//				builder2.WriteString(gcdString)
//			}
//			if builder1.String() != str1 || builder2.String() != str2 {
//				return ""
//			}
//			return gcdString
//		}
//	}
func numSteps(s string) int {
	bytes := []byte(s)
	cnt := 0
	for len(bytes) > 1 || bytes[0] != '1' {
		if bytes[len(bytes)-1] == '0' {
			bytes = bytes[:len(bytes)-1]
		} else {
			bytes[len(bytes)-1] = '0'
			carry := 1
			for i := len(bytes) - 2; i >= 0 && carry != 0; i-- {
				if bytes[i] == '1' {
					bytes[i] = '0'
				} else {
					bytes[i] = '1'
					carry = 0
				}
			}
			if carry != 0 {
				bytes = append([]byte{'1'}, bytes...)
			}
		}
		cnt++
	}
	return cnt
}
func canReach(arr []int, start int) bool {
	canReachArr := make([]int, 0, len(arr))
	for i := 0; i < len(arr); i++ {
		if arr[i] == 0 {
			if i == start {
				return true
			}
			canReachArr = append(canReachArr, i)
		}
	}
	flag := true
	for flag {
		flag = false
		for i := 0; i < len(arr); i++ {
			if !slices.Contains(canReachArr, i) {
				if (i-arr[i] >= 0 && slices.Contains(canReachArr, i-arr[i])) || (i+arr[i] < len(arr) && slices.Contains(canReachArr, i+arr[i])) {
					if i == start {
						return true
					}
					canReachArr = append(canReachArr, i)
					flag = true
				}
			}
		}
	}
	return false
}
func minDeletionSize(strs []string) int {
	arrs := make([][]int, len(strs[0]))
	for i := 0; i < len(strs); i++ {
		for j := 0; j < len(strs[i]); j++ {
			arrs[j] = append(arrs[j], int(strs[i][j]-'a'))
		}
	}
	ans := 0
	for _, arr := range arrs {
		if !slices.IsSorted(arr) {
			ans++
		}
	}
	return ans
}
func minMaxDifference(num int) int {
	str := strconv.Itoa(num)
	firstNoNine := strings.IndexFunc(str, func(r rune) bool {
		return r != '9'
	})
	var maxStr string
	if firstNoNine != -1 {
		maxStr = strings.ReplaceAll(str, str[firstNoNine:firstNoNine+1], "9")
	} else {
		maxStr = str
	}
	maxNum, _ := strconv.Atoi(maxStr)
	minStr := strings.ReplaceAll(str, str[0:1], "0")
	minNum, _ := strconv.Atoi(minStr)
	return maxNum - minNum
}
func maxRepeating(sequence string, word string) int {
	builder := strings.Builder{}
	for i := 0; i <= len(sequence)/len(word); i++ {
		builder.WriteString(word)
		if !strings.Contains(sequence, builder.String()) {
			return i
		}
	}
	return 0
}
func Abs(num int) int {
	if num >= 0 {
		return num
	} else {
		return -num
	}
}
func minimumAverageDifference(nums []int) int {
	prefix := make([]int, len(nums))
	sum := 0
	for i, num := range nums {
		sum += num
		prefix[i] = sum
	}
	minimum := math.MaxInt
	minimumIndex := -1
	for i := 0; i < len(nums); i++ {
		leftAverage := prefix[i] / (i + 1)
		var rightAverage int
		if i != len(nums)-1 {
			rightAverage = (sum - prefix[i]) / (len(nums) - i - 1)
		} else {
			rightAverage = 0
		}
		diff := Abs(leftAverage - rightAverage)
		if diff < minimum {
			minimum = diff
			minimumIndex = i
		}
	}
	return minimumIndex
}
func maxTurbulenceSize(arr []int) int {
	ans := 1
	pre := 0
	left := 0
	for i := 1; i < len(arr); i++ {
		if i != 1 {
			if (arr[i]-arr[i-1])*pre >= 0 {
				left = i - 1
			} else {
				ans = max(ans, i-left+1)
			}
		}
		pre = arr[i] - arr[i-1]
		if pre != 0 {
			ans = max(ans, 2)
		}
	}
	return ans
}

//	func minFlips(target string) int {
//		builder := strings.Builder{}
//		for i := 0; i < len(target); {
//			builder.WriteByte(target[i])
//			j := i + 1
//			for j < len(target) && target[j] == target[i] {
//				j++
//			}
//			i = j
//		}
//		ans := len(builder.String())
//		if builder.String()[0] == '0' {
//			return ans - 1
//		}
//		return ans
//	}
func prisonAfterNDays(cells []int, n int) []int {
	t := make([][]int, 14)
	for i := 0; i < len(t); i++ {
		var pre []int
		if i == 0 {
			pre = cells
		} else {
			pre = t[i-1]
		}
		t[i] = append(t[i], 0)
		for j := 1; j < 7; j++ {
			if pre[j-1] == pre[j+1] {
				t[i] = append(t[i], 1)
				t[i][j] = 1
			} else {
				t[i] = append(t[i], 0)
			}
		}
		t[i] = append(t[i], 0)
	}
	return t[(n-1)%14]
}
func queensAttacktheKing(queens [][]int, king []int) (ans [][]int) {
	minRowDist := []int{-100, 100}
	minColDist := []int{-100, 100}
	minDiag1Dist := []int{-100, 100}
	minDiag2Dist := []int{-100, 100}
	for _, queen := range queens {
		if queen[0] == king[0] {
			if queen[1] < king[1] {
				minRowDist[0] = max(minRowDist[0], queen[1]-king[1])
			} else {
				minRowDist[1] = min(minRowDist[1], queen[1]-king[1])
			}
		} else if queen[1] == king[1] {
			if queen[0] < king[0] {
				minColDist[0] = max(minColDist[0], queen[0]-king[0])
			} else {
				minColDist[1] = min(minColDist[1], queen[0]-king[0])
			}
		} else if queen[1]-queen[0] == king[1]-king[0] {
			if queen[0] < king[0] {
				minDiag1Dist[0] = max(minDiag1Dist[0], queen[0]-king[0])
			} else {
				minDiag1Dist[1] = min(minDiag1Dist[1], queen[0]-king[0])
			}
		} else if queen[0]+queen[1] == king[0]+king[1] {
			if queen[0] < king[0] {
				minDiag2Dist[0] = max(minDiag2Dist[0], queen[0]-king[0])
			} else {
				minDiag2Dist[1] = min(minDiag2Dist[1], queen[0]-king[0])
			}
		}
	}
	if minRowDist[0] != -100 {
		ans = append(ans, []int{king[0], king[1] + minRowDist[0]})
	}
	if minRowDist[1] != 100 {
		ans = append(ans, []int{king[0], king[1] + minRowDist[1]})
	}
	if minColDist[0] != -100 {
		ans = append(ans, []int{king[0] + minColDist[0], king[1]})
	}
	if minColDist[1] != 100 {
		ans = append(ans, []int{king[0] + minColDist[1], king[1]})
	}
	if minDiag1Dist[0] != -100 {
		ans = append(ans, []int{king[0] + minDiag1Dist[0], king[1] + minDiag1Dist[0]})
	}
	if minDiag1Dist[1] != 100 {
		ans = append(ans, []int{king[0] + minDiag1Dist[1], king[1] + minDiag1Dist[1]})
	}
	if minDiag2Dist[0] != -100 {
		ans = append(ans, []int{king[0] + minDiag2Dist[0], king[1] - minDiag2Dist[0]})
	}
	if minDiag2Dist[1] != 100 {
		ans = append(ans, []int{king[0] + minDiag2Dist[1], king[1] - minDiag2Dist[1]})
	}
	return
}
