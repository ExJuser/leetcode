package bianry

import (
	"slices"
)

// 35. 搜索插入位置
func searchInsert(nums []int, target int) int {
	left, right := 0, len(nums)-1
	ans := len(nums)
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] >= target {
			right = mid - 1
			ans = mid
		} else {
			left = mid + 1
		}
	}
	return ans
}

// 744. 寻找比目标字母大的最小字母
func nextGreatestLetter(letters []byte, target byte) byte {
	var ans int
	left, right := 0, len(letters)-1
	for left <= right {
		mid := (right-left)/2 + left
		if letters[mid] > target {
			ans = mid
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return letters[ans]
}

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
func smallestDivisor(nums []int, threshold int) int {
	var check func(divisor int) bool
	check = func(divisor int) bool {
		var ans int
		for i := 0; i < len(nums); i++ {
			ans += (nums[i] + divisor - 1) / divisor
		}
		return ans <= threshold
	}
	left, right := 1, slices.Max(nums)
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

// 2187. 完成旅途的最少时间
func minimumTime(time []int, totalTrips int) int64 {
	//在i单位时间内能否完成totalTrips次旅行
	var check func(i int) bool
	check = func(i int) bool {
		var trips int
		for _, t := range time {
			trips += i / t
		}
		return trips >= totalTrips
	}
	left, right := 0, slices.Min(time)*totalTrips
	ans := right
	for left <= right {
		mid := (right-left)/2 + left
		if check(mid) {
			right = mid - 1
			ans = mid
		} else {
			left = mid + 1
		}
	}
	return int64(ans)
}

// 2226. 每个小孩最多能分到多少糖果 每个小孩分到的数量要相同
func maximumCandies(candies []int, k int64) int {
	//每个小孩分到i颗糖果能否满足要求
	var check func(i int) bool
	check = func(i int) bool {
		var cnt int64
		//遍历糖果堆 看每一堆能满足多少孩子
		for _, candy := range candies {
			cnt += int64(candy / i)
			if cnt >= k {
				return true
			}
		}
		return cnt >= k
	}
	left, right := 1, slices.Max(candies)
	ans := 0
	for left <= right {
		mid := (right-left)/2 + left
		if check(mid) {
			left = mid + 1
			ans = mid
		} else {
			right = mid - 1
		}
	}
	return ans
}
