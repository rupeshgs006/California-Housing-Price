
from locust import HttpUser, task, constant, events
from locust.runners import MasterRunner
import gevent

class HousingAPIUser(HttpUser):
    wait_time = constant(0)

    @task
    def predict_house_price(self):
        payload = {
            "MedInc": 8.3,
            "HouseAge": 25,
            "AveRooms": 6.5,
            "AveBedrms": 1.1,
            "Population": 1200,
            "AveOccup": 3.0,
            "Latitude": 37.8,
            "Longitude": -122.4,
        }

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
        ) as response:

            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}"
                )
                return

            try:
                result = response.json()
            except ValueError:
                response.failure("Invalid JSON response")
                return

            if (
                "predicted_price" not in result
                or "currency" not in result
            ):
                response.failure("Unexpected response format")
            else:
                response.success()

@events.test_start.add_listener
def stop_after_request_limit(environment, **kwargs):
    runner = environment.runner
    if runner is None or isinstance(runner, MasterRunner):
        return

    completed_requests = 0

    def count_requests(name, **kwargs):
        nonlocal completed_requests
        if name != "/predict":
            return
        completed_requests += 1
        if completed_requests == 10000:
            gevent.spawn(runner.quit)

    def remove_listeners(**kwargs):
        environment.events.request.remove_listener(count_requests)
        environment.events.test_stop.remove_listener(remove_listeners)

    environment.events.request.add_listener(count_requests)
    environment.events.test_stop.add_listener(remove_listeners)
