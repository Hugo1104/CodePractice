import java.util.*;

public class SupermarketCheckout {

    private static class Customer {
        int id;
        int items;

        Customer(int id, int items) {
            this.id = id;
            this.items = items;
        }
    }

    private List<Queue<Customer>> lines;

    public SupermarketCheckout(int n) {
        lines = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            lines.add(new LinkedList<>());
        }
    }

    public void customerEnter(int customerId, int lineNumber, int numItems) {
        lines.get(lineNumber).add(new Customer(customerId, numItems));
    }

    public void basketChange(int customerId, int newNumItems) {
        for (Queue<Customer> line : lines) {
            for (Customer customer : line) {
                if (customer.id == customerId) {
                    customer.items = newNumItems;
                    break;
                }
            }
        }
    }

    public List<Integer> lineService(int lineNumber, int numProcessedItems) {
        List<Integer> output = new ArrayList<>();
        Queue<Customer> line = lines.get(lineNumber);

        while (!line.isEmpty() && numProcessedItems > 0) {
            Customer customer = line.peek();

            if (customer.items <= numProcessedItems) {
                numProcessedItems -= customer.items;
                output.add(line.poll().id);
            } else {
                customer.items -= numProcessedItems;
                numProcessedItems = 0;
            }
        }

        return output;
    }

    public List<Integer> lineService() {
        List<Integer> output = new ArrayList<>();

        for (int i = 0; i < lines.size(); i++) {
            if (!lines.get(i).isEmpty()) {
                output.addAll(lineService(i, 1));
            }
        }

        return output;
    }

    public static void main(String[] args) {
        SupermarketCheckout checkout = new SupermarketCheckout(5);
        checkout.customerEnter(123, 1, 5);
        checkout.customerEnter(2, 2, 3);
        checkout.customerEnter(3, 1, 2);
        System.out.println(checkout.lineService(1, 6)); // Expected: [123, 3]

        checkout.customerEnter(123, 1, 5);
        checkout.customerEnter(3, 1, 2);
        checkout.basketChange(123, 6);
        System.out.println(checkout.lineService(1, 4)); // Expected: [3]
        System.out.println(checkout.lineService(1, 5)); // Expected: [123]
    }
}
