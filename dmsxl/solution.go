package dmsxl

// 704. 二分查找
func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}

// 35. 搜索插入位置
func searchInsert(nums []int, target int) int {
	left, right := 0, len(nums)-1
	ans := len(nums)
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			ans = mid
			right = mid - 1
		}
	}
	return ans
}

// 34. 在排序数组中查找元素的第一个和最后一个位置
func searchRange(nums []int, target int) []int {
	pivot := search(nums, target)
	if pivot == -1 {
		return []int{-1, -1}
	}
	left, right := pivot, pivot
	for left >= 0 && nums[left] == target {
		left--
	}
	for right < len(nums) && nums[right] == target {
		right++
	}
	return []int{left + 1, right - 1}
}

// 69. x 的平方根
func mySqrt(x int) int {
	left, right := 0, x
	var ans int
	for left <= right {
		mid := (right-left)/2 + left
		if mid*mid > x {
			right = mid - 1
		} else {
			ans = mid
			left = mid + 1
		}
	}
	return ans
}

// 367. 有效的完全平方数
func isPerfectSquare(num int) bool {
	left, right := 1, num
	for left <= right {
		mid := (right-left)/2 + left
		square := mid * mid
		if square == num {
			return true
		} else if square < num {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return false
}

// 27. 移除元素
func removeElement(nums []int, val int) int {
	end := len(nums) - 1
	for i := 0; i <= end; {
		if nums[i] == val {
			nums[i], nums[end] = nums[end], nums[i]
			end--
		} else {
			i++
		}
	}
	return end + 1
}

// 26. 删除有序数组中的重复项
func removeDuplicates(nums []int) int {
	var count int
	for i := 0; i < len(nums); i++ {
		if i == 0 || nums[i] != nums[i-1] {
			nums[count] = nums[i]
			count++
		}
	}
	return count
}

// 283. 移动零
func moveZeroes(nums []int) {
	index := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[index] = nums[i]
			index++
		}
	}
	nums = append(nums[:index], make([]int, len(nums)-index)...)
}

// 844. 比较含退格的字符串
func backspaceCompare(s string, t string) bool {
	var help func(str string) string
	help = func(str string) string {
		bytes := []byte(str)
		index := 0
		for i := 0; i < len(bytes); i++ {
			if bytes[i] == '#' {
				index = max(index-1, 0)
			} else {
				bytes[index] = bytes[i]
				index++
			}
		}
		return string(bytes[:index])
	}
	return help(s) == help(t)
}
