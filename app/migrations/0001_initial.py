# -*- coding: utf-8 -*-
# Generated by Django 1.9 on 2019-03-03 05:39
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Client',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('monthly_budget', models.CharField(max_length=10)),
                ('monthly_income', models.CharField(max_length=10)),
                ('r1', models.IntegerField(default=1)),
                ('r2', models.IntegerField(default=1)),
                ('r3', models.IntegerField(default=1)),
                ('r4', models.IntegerField(default=1)),
                ('r5', models.IntegerField(default=1)),
                ('r6', models.IntegerField(default=1)),
                ('risk_quotient', models.IntegerField(blank=True, default=10)),
                ('type_of_investor', models.CharField(blank=True, max_length=100)),
                ('stocks', models.CharField(blank=True, max_length=250)),
                ('cryptos', models.CharField(blank=True, max_length=250)),
            ],
        ),
    ]
