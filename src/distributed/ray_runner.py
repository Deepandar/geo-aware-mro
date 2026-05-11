try:
    import ray

    RAY_AVAILABLE = True

except Exception:

    RAY_AVAILABLE = False

def ray_status():

    return {
        "ray_available": RAY_AVAILABLE
    }
