import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class CompanyExercise {
    public int strStr(String source, String target) {
        // Write your code here
        int n = source.length(), m = target.length();

        for (int i = 0; i <= n - m; i++) {
            int start = i, index = 0;

            while (index < m && source.charAt(start) == target.charAt(index)) {
                start++;
                index++;
            }

            if (index == m) {
                return i;
            }
        }

        return -1;
    }

    public int[] twoSum(int[] numbers, int target) {
        // write your code here
        if (numbers.length == 0) {
            return new int[]{-1, -1};
        }

        Map<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < numbers.length; i++) {
            map.put(numbers[i], i);
        }

        for (int i = 0; i < numbers.length; i++) {
            int num = numbers[i];
            if (map.containsKey(target - num) && map.get(target - num) != i) {
                int[] ans = new int[] {i, map.get(target - num)};
                Arrays.sort(ans);
                return ans;
            }
        }

        return new int[]{-1, -1};
    }
}
