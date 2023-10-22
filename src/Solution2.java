import java.util.*;
import java.text.DecimalFormat;

public class Solution2 {
    public class TruckPosition {
        public double mX;
        public double mY;
    }

    public class TruckPositionDelta {
        public int mTruckId;
        public double mDeltaX;
        public double mDeltaY;
    }

    public interface IServer {
        public TruckPosition SubscribeToTruck(int truckId);
    }

    public interface ISubscriber {
        // Called by server after initial subscription
        public void ProcessUpdate(final TruckPositionDelta positionDelta);

        // Called by clients
        public TruckPosition SubscribeToTruck(int truckId, int clientId);
        public List<TruckPositionDelta> GetUpdates(int clientId);
    }

    class Subscriber implements ISubscriber {
        private final IServer mServer;
        private Map<Integer, TruckPosition> mCurrentTruckPositions = new HashMap<>();
        private Map<Integer, List<TruckPositionDelta>> mClientUpdates = new HashMap<>();

        public Subscriber(IServer server) {
            mServer = server;
        }

        @Override
        public void ProcessUpdate(final TruckPositionDelta positionDelta) {
            // Process the update and store for each subscribed client
            List<Integer> subscribedClients = ((Server)mServer).getClientsSubscribedTo(positionDelta.mTruckId);
            for (int clientId : subscribedClients) {
                mClientUpdates.computeIfAbsent(clientId, k -> new ArrayList<>()).add(positionDelta);
            }

            // Update the current position of the truck in the subscriber
            TruckPosition position = mCurrentTruckPositions.get(positionDelta.mTruckId);
            if (position == null) {
                position = new TruckPosition();
                mCurrentTruckPositions.put(positionDelta.mTruckId, position);
            }
            position.mX += positionDelta.mDeltaX;
            position.mY += positionDelta.mDeltaY;
        }

        @Override
        public TruckPosition SubscribeToTruck(int truckId, int clientId) {
            // Subscribe client to the truck
            ((Server)mServer).addSubscriber(clientId, truckId);

            // Get the current position of the truck
            TruckPosition position = mServer.SubscribeToTruck(truckId);
            mCurrentTruckPositions.put(truckId, position);

            return position;
        }

        @Override
        public List<TruckPositionDelta> GetUpdates(int clientId) {
            // Return all updates for the client and clear it
            List<TruckPositionDelta> updates = mClientUpdates.remove(clientId);
            if (updates == null) {
                return new ArrayList<TruckPositionDelta>();
            }
            return updates;
        }
    }


    class Server implements IServer {
        private HashSet<Integer> mRegisteredTrucks;
        private HashMap<Integer, TruckPosition> mCurrentPos;
        private Map<Integer, List<Integer>> truckSubscribers= new HashMap<>();
        private Map<Integer, List<Integer>> clientSubscriptions = new HashMap<>();

        public Server() {
            mRegisteredTrucks = new HashSet<>();
            mCurrentPos = new HashMap<>();
        }

        @Override
        public TruckPosition SubscribeToTruck(int truckId) {
            mRegisteredTrucks.add(truckId);
            TruckPosition pos = mCurrentPos.get(truckId);
            TruckPosition copy = new TruckPosition();
            copy.mX = pos.mX;
            copy.mY = pos.mY;
            return copy;
        }

        public void AddPosition(int truckId, TruckPosition pos) {
            mCurrentPos.put(truckId, pos);
        }

        public void OnUpdate(Subscriber subscriber, final TruckPositionDelta delta) {
            if (mRegisteredTrucks.contains(delta.mTruckId))
            {
                subscriber.ProcessUpdate(delta);
            }
            TruckPosition pos = mCurrentPos.get(delta.mTruckId);
            pos.mX += delta.mDeltaX;
            pos.mY += delta.mDeltaY;
        }

        public List<Integer> getClientsSubscribedTo(int truckId) {
            return truckSubscribers.getOrDefault(truckId, new ArrayList<>());
        }

        public void addSubscriber(int clientId, int truckId) {
            truckSubscribers.computeIfAbsent(truckId, k -> new ArrayList<>()).add(clientId);
            clientSubscriptions.computeIfAbsent(clientId, k -> new ArrayList<>()).add(truckId);
        }
    }

    class Client {
        private final int mClientId;
        private final Subscriber mSubscriber;
        private final DecimalFormat mFormat;

        public Client(int clientId, Subscriber subscriber) {
            mClientId = clientId;
            mSubscriber = subscriber;
            mFormat = new DecimalFormat("0.#");
        }

        public void Subscribe(int truckId) {
            TruckPosition pos = mSubscriber.SubscribeToTruck(truckId, mClientId);
            System.out.println("S " + mClientId + " " + truckId + " " + mFormat.format(pos.mX) + " " + mFormat.format(pos.mY));
        }

        public void RequestUpdate() {
            List<TruckPositionDelta> updates = mSubscriber.GetUpdates(mClientId);
            for (final TruckPositionDelta delta : updates) {
                System.out.println("U " + mClientId + " " + delta.mTruckId + " " + mFormat.format(delta.mDeltaX) + " " + mFormat.format(delta.mDeltaY));
            }
        }
    }

    public static void main(String[] args) {
        Solution2 solution = new Solution2();
        Server server = solution.new Server();
        Subscriber subscriber = solution.new Subscriber(server);
        List<Client> clients = new ArrayList<>();

        Scanner scanner = new Scanner(System.in);
        int numTrucks = scanner.nextInt();
        for (int i = 0; i < numTrucks; i++) {
            TruckPosition pos = solution.new TruckPosition();
            pos.mX = scanner.nextDouble();
            pos.mY = scanner.nextDouble();
            server.AddPosition(i, pos);
        }

        while (scanner.hasNext()) {
            char command = scanner.next().charAt(0);
            if (command == 'S') {
                int clientId = scanner.nextInt();
                if (clientId >= clients.size()) {
                    clients.add(solution.new Client(clientId, subscriber));
                }
                clients.get(clientId).Subscribe(scanner.nextInt());
            } else if (command == 'U') {
                TruckPositionDelta delta = solution.new TruckPositionDelta();
                delta.mTruckId = scanner.nextInt();
                delta.mDeltaX = scanner.nextDouble();
                delta.mDeltaY = scanner.nextDouble();
                server.OnUpdate(subscriber, delta);
            } else if (command == 'R') {
                int clientId = scanner.nextInt();
                clients.get(clientId).RequestUpdate();
            } else {
                throw new IllegalArgumentException("Invalid input");
            }
        }
    }
}