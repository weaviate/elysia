import datetime

def create_reference(date: datetime.datetime = datetime.datetime.now()) -> str:
    return {
        "datetime": date.isoformat(),
        "day_of_week": date.strftime("%A"),
        "time_of_day": date.strftime("%I:%M %p")
    }

reference = create_reference()