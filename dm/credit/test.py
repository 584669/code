def par(nums,l,r):
    temp=nums[r]
    j=l
    for i in range(l,r):
        if(nums[i]<temp):
            nums[i],nums[j]=nums[j],nums[i]
            j+=1
    nums[r], nums[j] = nums[j], nums[r]
    return j
def get(nums,l,r):
    if(l<r):
        mid=par(nums,l,r)
        get(nums,l,mid-1)
        get(nums,mid+1,r)
def quit_sort(nums):
    get(nums,0,len(nums)-1)
    return nums
test=[4,3,2,1]
print(quit_sort(test))