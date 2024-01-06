public class ThreeStacks {
    int size;
    int[] arr;
    int[] pos = new int[3];

    /*
     * @param size: An integer
     */public ThreeStacks(int size) {
        // do initialization if necessary
        arr = new int[3 * size];
        pos[0] = 0;
        pos[1] = size;
        pos[2] = 2 * size;
        this.size = size;
    }

    /*
     * @param stackNum: An integer
     * @param value: An integer
     * @return: nothing
     */
    public void push(int stackNum, int value) {
        // Push value into stackNum stack
        arr[pos[stackNum]++] = value;
    }

    /*
     * @param stackNum: An integer
     * @return: the top element
     */
    public int pop(int stackNum) {
        // Pop and return the top element from stackNum stack
        return arr[--pos[stackNum]];
    }

    /*
     * @param stackNum: An integer
     * @return: the top element
     */
    public int peek(int stackNum) {
        // Return the top element
        return arr[pos[stackNum] - 1];
    }

    /*
     * @param stackNum: An integer
     * @return: true if the stack is empty else false
     */
    public boolean isEmpty(int stackNum) {
        // write your code here
        return pos[stackNum] == (size * stackNum);
    }
}