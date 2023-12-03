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
}
