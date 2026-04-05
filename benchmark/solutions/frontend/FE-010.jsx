import React, { useState } from 'react';

const steps = [
  ['name', 'email'],
  ['address', 'city']
];

export default function FormWizard({ onSubmit = () => {} }) {
  const [step, setStep] = useState(1);
  const [values, setValues] = useState({
    name: '',
    email: '',
    address: '',
    city: ''
  });
  const [errors, setErrors] = useState({});

  const validateStep = () => {
    const fields = steps[step - 1] || [];
    const nextErrors = {};

    fields.forEach((field) => {
      if (!values[field].trim()) {
        nextErrors[field] = 'This field is required';
      }
    });

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const updateField = (field, value) => {
    setValues((currentValues) => ({
      ...currentValues,
      [field]: value
    }));
    setErrors((currentErrors) => ({
      ...currentErrors,
      [field]: ''
    }));
  };

  const goToNextStep = () => {
    if (!validateStep()) {
      return;
    }

    setStep((currentStep) => currentStep + 1);
  };

  const handleSubmit = () => {
    onSubmit(values);
  };

  return (
    <div>
      <p>Step {step} of 3</p>

      {step === 1 ? (
        <div>
          <label htmlFor="wizard-name">Name</label>
          <input
            id="wizard-name"
            value={values.name}
            onChange={(event) => updateField('name', event.target.value)}
          />
          {errors.name ? <p>{errors.name}</p> : null}

          <label htmlFor="wizard-email">Email</label>
          <input
            id="wizard-email"
            type="email"
            value={values.email}
            onChange={(event) => updateField('email', event.target.value)}
          />
          {errors.email ? <p>{errors.email}</p> : null}
        </div>
      ) : null}

      {step === 2 ? (
        <div>
          <label htmlFor="wizard-address">Address</label>
          <input
            id="wizard-address"
            value={values.address}
            onChange={(event) => updateField('address', event.target.value)}
          />
          {errors.address ? <p>{errors.address}</p> : null}

          <label htmlFor="wizard-city">City</label>
          <input
            id="wizard-city"
            value={values.city}
            onChange={(event) => updateField('city', event.target.value)}
          />
          {errors.city ? <p>{errors.city}</p> : null}
        </div>
      ) : null}

      {step === 3 ? (
        <div>
          <h2>Review</h2>
          <p>Name: {values.name}</p>
          <p>Email: {values.email}</p>
          <p>Address: {values.address}</p>
          <p>City: {values.city}</p>
        </div>
      ) : null}

      {step > 1 ? (
        <button type="button" onClick={() => setStep((currentStep) => currentStep - 1)}>
          Previous
        </button>
      ) : null}

      {step < 3 ? (
        <button type="button" onClick={goToNextStep}>
          Next
        </button>
      ) : (
        <button type="button" onClick={handleSubmit}>
          Submit
        </button>
      )}
    </div>
  );
}

