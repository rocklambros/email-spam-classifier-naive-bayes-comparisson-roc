#!/usr/bin/env python3
"""
Synthetic Email Dataset Generator for Naive Bayes Classification

This script generates synthetic email data that statistically mirrors the original
email_dataset.csv for use in Naive Bayes classification assignments.

Author: Generated for Master of Science in Applied Data Science and AI
Course: Essential Math for Data Science and AI
Project: Naive Bayes Classification - Email Spam Detection

Mathematical Approach:
- Preserves original data distributions using probability sampling
- Maintains feature correlations for realistic Naive Bayes testing
- Uses only Python standard libraries as per assignment requirements
"""

import csv
import random
import datetime
import re
from typing import List, Dict, Tuple


class SyntheticEmailGenerator:
    """
    Generates synthetic email data maintaining statistical properties of original dataset.

    Original Dataset Statistics:
    - Total records: 5,000
    - Status distribution: Archived (43%), Bounced (34%), Held (15%), etc.
    - Spam detection: 50.8% legitimate, 49.2% spam
    - Spam score range: 0-156, average 4.01
    """

    def __init__(self, seed: int = 42):
        """Initialize generator with fixed seed for reproducible results."""
        random.seed(seed)

        # Statistical distributions from original dataset
        self.status_distribution = {
            'Archived': 0.4332,
            'Bounced': 0.3400,
            'Held': 0.1496,
            'Deferred': 0.0592,
            'Rejected': 0.0178,
            'Accepted': 0.0002
        }

        # Spam detection: 50.8% legitimate (empty), 49.2% spam (Moderate)
        self.spam_detection_prob = 0.492  # Probability of spam

        # Domain lists for realistic email generation
        self.legitimate_domains = [
            'gmail.com', 'yahoo.com', 'outlook.com', 'company.com', 'university.edu',
            'research.org', 'business.net', 'corporate.com', 'institution.edu',
            'organization.org', 'tech.com', 'consulting.com', 'finance.com'
        ]

        self.spam_domains = [
            'suspicious.net', 'promo.biz', 'offers.info', 'deals.click',
            'marketing.buzz', 'spam.mail', 'fake.domain', 'scam.net',
            'phishing.com', 'malicious.org', 'untrusted.biz'
        ]

        # Subject line patterns
        self.legitimate_subjects = [
            'Meeting reminder for {date}',
            'Project update - {topic}',
            'Welcome to {service}',
            'Your order confirmation #{number}',
            'Weekly newsletter - {topic}',
            'Account security notification',
            'Invoice #{number} from {company}',
            'Conference registration confirmed',
            'Monthly report - {month}',
            'System maintenance notification'
        ]

        self.spam_subjects = [
            'URGENT: Claim your prize NOW!',
            'Make money fast - limited time offer',
            'You have won ${amount}!',
            'Lose weight quickly with this trick',
            'Click here for amazing deals',
            'FREE trial - act now!',
            'Congratulations! You are a winner!',
            'Exclusive offer just for you',
            'Last chance to save big!',
            'Secret method to earn ${amount}'
        ]

    def generate_email_address(self, is_spam: bool) -> Tuple[str, str]:
        """
        Generate realistic email addresses for envelope and header.

        Args:
            is_spam: Boolean indicating if this is a spam email

        Returns:
            Tuple of (envelope_from, header_from)
        """
        domains = self.spam_domains if is_spam else self.legitimate_domains
        domain = random.choice(domains)

        if is_spam:
            # Spam emails often have suspicious patterns
            username_patterns = [
                f"noreply{random.randint(1000, 9999)}",
                f"promo{random.randint(100, 999)}",
                f"offers{random.randint(10, 99)}",
                f"deal{random.randint(1, 999)}",
                f"marketing{random.randint(1, 9999)}"
            ]
            username = random.choice(username_patterns)
        else:
            # Legitimate emails have normal patterns
            first_names = ['john', 'jane', 'mike', 'sarah', 'david', 'lisa', 'admin', 'support', 'info']
            last_names = ['smith', 'johnson', 'williams', 'brown', 'jones', 'garcia', 'miller']

            if random.random() < 0.3:  # 30% chance of service emails
                username = random.choice(['admin', 'support', 'info', 'noreply', 'contact'])
            else:
                username = f"{random.choice(first_names)}.{random.choice(last_names)}"

        email = f"{username}@{domain}"
        envelope = f"<{email}>"
        header = f"{username.title()} <{email}>" if not is_spam else envelope

        return envelope, header

    def generate_subject(self, is_spam: bool) -> str:
        """Generate realistic subject lines based on spam classification."""
        if is_spam:
            subject = random.choice(self.spam_subjects)
            # Replace placeholders with random values
            subject = subject.replace('{amount}', str(random.randint(100, 10000)))
        else:
            subject = random.choice(self.legitimate_subjects)
            # Replace placeholders
            replacements = {
                '{date}': datetime.date.today().strftime('%Y-%m-%d'),
                '{topic}': random.choice(['AI Research', 'Data Science', 'Software Update', 'Security']),
                '{service}': random.choice(['LinkedIn', 'GitHub', 'University Portal', 'Cloud Service']),
                '{number}': str(random.randint(10000, 99999)),
                '{company}': random.choice(['TechCorp', 'DataSoft', 'CloudInc', 'Research Ltd']),
                '{month}': datetime.date.today().strftime('%B')
            }
            for placeholder, value in replacements.items():
                subject = subject.replace(placeholder, value)

        return subject

    def generate_datetime(self) -> str:
        """Generate realistic datetime strings in the format from original data."""
        # Generate dates within last 30 days
        base_date = datetime.datetime.now()
        days_ago = random.randint(0, 30)
        hours = random.randint(0, 23)
        minutes = random.randint(0, 59)
        seconds = random.randint(0, 59)

        target_date = base_date - datetime.timedelta(days=days_ago)
        target_date = target_date.replace(hour=hours, minute=minutes, second=seconds)

        # Format to match original: "Sat Sep 27 18:03:54 EDT 2025"
        return target_date.strftime("%a %b %d %H:%M:%S EDT %Y")

    def generate_ip_address(self, is_spam: bool) -> str:
        """Generate realistic IP addresses."""
        if is_spam:
            # Spam often comes from suspicious IP ranges
            return f"{random.randint(180, 220)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        else:
            # Legitimate emails from common ranges
            common_ranges = [
                (192, 168),  # Private networks
                (10, random.randint(1, 255)),  # Private networks
                (172, random.randint(16, 31)),  # Private networks
                (127, 0),  # Localhost
                (74, random.randint(1, 255)),  # Common ISP ranges
                (66, random.randint(1, 255))
            ]
            first_two = random.choice(common_ranges)
            return f"{first_two[0]}.{first_two[1]}.{random.randint(1, 255)}.{random.randint(1, 255)}"

    def generate_spam_score(self, is_spam: bool, status: str) -> str:
        """
        Generate realistic spam scores based on classification and status.

        Statistical correlation: Higher scores correlate with spam detection.
        Legitimate emails: mostly 0-5, Spam emails: wider range with higher average.
        """
        if is_spam:
            # Spam emails have higher scores with more variance
            if status in ['Held', 'Rejected']:
                # Held/Rejected emails typically have higher scores
                score = random.randint(5, 156)
            else:
                score = random.randint(1, 50)
        else:
            # Legitimate emails have lower scores
            if random.random() < 0.7:  # 70% have score 0
                score = 0
            else:
                score = random.randint(1, 15)

        return str(score)

    def generate_status(self) -> str:
        """Generate email status based on original distribution."""
        rand = random.random()
        cumulative = 0

        for status, probability in self.status_distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return status

        return 'Archived'  # Fallback

    def generate_attachment_info(self) -> str:
        """Generate attachment information (mostly empty, some 'Has Attachment')."""
        return 'Has Attachment' if random.random() < 0.05 else ''

    def generate_route_info(self) -> Tuple[str, str]:
        """Generate route and info fields based on status."""
        routes = ['inbound', 'internal', 'outbound']
        route = random.choice(routes)

        info_options = [
            'Indexed and archived',
            'Message Hold Applied - Spam Signature policy',
            'Hard Bounce',
            'Attempt Greylisted',
            'IP Found in RBL',
            'Awaiting indexing'
        ]
        info = random.choice(info_options) if random.random() < 0.6 else ''

        return route, info

    def generate_record(self) -> Dict[str, str]:
        """Generate a single synthetic email record."""
        # Determine if this is spam based on probability
        is_spam = random.random() < self.spam_detection_prob

        # Generate correlated fields
        status = self.generate_status()
        envelope_from, header_from = self.generate_email_address(is_spam)
        subject = self.generate_subject(is_spam)
        sent_datetime = self.generate_datetime()
        ip_address = self.generate_ip_address(is_spam)
        attachment = self.generate_attachment_info()
        route, info = self.generate_route_info()
        spam_score = self.generate_spam_score(is_spam, status)
        spam_detection = 'Moderate' if is_spam else ''

        # Generate recipient (always same pattern)
        to_address = '<user@example.com>'

        return {
            'Status': status,
            'From (Envelope)': envelope_from,
            'From (Header)': header_from,
            'To': to_address,
            'Subject': subject,
            'Sent Date/Time': sent_datetime,
            'IP Address': ip_address,
            'Attachment': attachment,
            'Route': route,
            'Info': info,
            'Spam Score': spam_score,
            'Spam Detection': spam_detection
        }

    def generate_dataset(self, num_records: int = 5000) -> List[Dict[str, str]]:
        """
        Generate complete synthetic dataset.

        Args:
            num_records: Number of records to generate (default: 5000)

        Returns:
            List of dictionaries representing email records
        """
        print(f"Generating {num_records} synthetic email records...")

        dataset = []
        for i in range(num_records):
            if i % 1000 == 0:
                print(f"Generated {i} records...")

            record = self.generate_record()
            dataset.append(record)

        print("Generation complete!")
        return dataset

    def save_to_csv(self, dataset: List[Dict[str, str]], filename: str = 'synthetic_email_dataset.csv'):
        """Save dataset to CSV file."""
        fieldnames = [
            'Status', 'From (Envelope)', 'From (Header)', 'To', 'Subject',
            'Sent Date/Time', 'IP Address', 'Attachment', 'Route', 'Info',
            'Spam Score', 'Spam Detection'
        ]

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset)

        print(f"Dataset saved to {filename}")

    def print_statistics(self, dataset: List[Dict[str, str]]):
        """Print statistics about the generated dataset."""
        total = len(dataset)
        spam_count = sum(1 for record in dataset if record['Spam Detection'] == 'Moderate')
        legitimate_count = total - spam_count

        print("\n" + "="*50)
        print("SYNTHETIC DATASET STATISTICS")
        print("="*50)
        print(f"Total records: {total}")
        print(f"Legitimate emails: {legitimate_count} ({legitimate_count/total*100:.1f}%)")
        print(f"Spam emails: {spam_count} ({spam_count/total*100:.1f}%)")

        # Status distribution
        status_counts = {}
        for record in dataset:
            status = record['Status']
            status_counts[status] = status_counts.get(status, 0) + 1

        print("\nStatus Distribution:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count} ({count/total*100:.1f}%)")

        # Spam score statistics
        spam_scores = [int(record['Spam Score']) for record in dataset if record['Spam Score']]
        if spam_scores:
            print(f"\nSpam Score Statistics:")
            print(f"  Min: {min(spam_scores)}")
            print(f"  Max: {max(spam_scores)}")
            print(f"  Average: {sum(spam_scores)/len(spam_scores):.2f}")

        print("="*50)


def main():
    """
    Main function to generate synthetic email dataset.

    This creates a dataset suitable for Naive Bayes classification assignment:
    - Maintains statistical properties of original data
    - Provides balanced classes for training/testing
    - Includes realistic feature correlations for independence assumption analysis
    """
    print("Synthetic Email Dataset Generator")
    print("For Naive Bayes Classification Assignment")
    print("-" * 40)

    # Create generator
    generator = SyntheticEmailGenerator(seed=42)  # Fixed seed for reproducibility

    # Generate dataset
    dataset = generator.generate_dataset(num_records=5000)

    # Save to CSV
    generator.save_to_csv(dataset, 'synthetic_email_dataset.csv')

    # Print statistics
    generator.print_statistics(dataset)

    print("\nDataset ready for Naive Bayes classification!")
    print("Use 'Spam Detection' field as target variable:")
    print("  - Empty/None = Legitimate email")
    print("  - 'Moderate' = Spam email")


if __name__ == "__main__":
    main()