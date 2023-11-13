public class AmazonPractice {

    public ListNode reverse(ListNode head) {
        // write your code here
        ListNode curr = head;
        ListNode prev = null;

        while (curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }

        return prev;
    }


    public String reverseString(String s) {
        // write your code here
        StringBuilder builder = new StringBuilder();

        for (int i = s.length() - 1; i >= 0; i--) {
            builder.append(s.charAt(i));
        }

        return builder.toString();
    }

}



