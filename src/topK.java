import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;
import java.util.stream.Collectors;

public class topK {
    /*
     * @param k: An integer
     */
    public int k;
    public PriorityQueue<Integer> minHeap;

    public topK(int k) {
        // do initialization if necessary
        this.k = k;
        this.minHeap = new PriorityQueue<>();
    }

    /*
     * @param num: Number to be added
     * @return: nothing
     */
    public void add(int num) {
        // write your code here
        minHeap.offer(num);
        if (minHeap.size() > k) {
            minHeap.poll();
        }
    }

    /*
     * @return: Top k element
     */
    public List<Integer> topk() {
        // write your code here

        int size = minHeap.size();

        int[] ans = new int[size];

        for (int i = size - 1; i >= 0; i--) {
            ans[i] = minHeap.poll();
        }

        for (int i = 0; i < size; i++) {
            minHeap.add(ans[i]);
        }

        return Arrays.stream(ans).boxed().collect(Collectors.toList());
    }
}