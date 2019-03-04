from django.db import models

# Create your models here.

class Client(models.Model):
    name = models.CharField(max_length=100)
    monthly_budget = models.CharField(max_length=10)
    monthly_income = models.CharField(max_length=10)

    # Risk Facts
    r1 = models.IntegerField(default=1)
    r2 = models.IntegerField(default=1)
    r3 = models.IntegerField(default=1)
    r4 = models.IntegerField(default=1)
    r5 = models.IntegerField(default=1)
    r6 = models.IntegerField(default=1)
    risk_quotient = models.IntegerField(blank=True, default=10)
    type_of_investor = models.CharField(max_length=100, blank=True)

    # Investments
    stocks = models.CharField(max_length=250, blank=True)
    cryptos = models.CharField(max_length=250, blank=True)

    def __str__(self):
        return self.name

# Create your models here.
