import java.util.*;

public class RandomizedCollection {

    Map<Integer, Set<Integer>> index;
    List<Integer> nums;
    /** Initialize your data structure here. */
    public RandomizedCollection() {
        index = new HashMap<>();
        nums = new ArrayList<>();
    }

    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    public boolean insert(int val) {
        // write your code here
        nums.add(val);
        Set<Integer> set = index.getOrDefault(val, new HashSet<>());
        set.add(nums.size() - 1);
        index.put(val, set);
        return set.size() == 1;
    }

    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    public boolean remove(int val) {
        // write your code here
        if (!index.containsKey(val)) {
            return false;
        }

        Iterator<Integer> iter = index.get(val).iterator();
        int i = iter.next();
        int lastNum = nums.get(nums.size() - 1);
        nums.set(i, lastNum);
        index.get(val).remove(i);
        index.get(lastNum).remove(nums.size() - 1);
        if (i < nums.size() - 1) {
            index.get(lastNum).add(i);
        }
        if (index.get(val).size() == 0) {
            index.remove(val);
        }
        nums.remove(nums.size() - 1);
        return true;
    }

    /** Get a random element from the collection. */
    public int getRandom() {
        // write your code here
        return nums.get((int) (Math.random() * nums.size()));
    }
}