# frozen_string_literal: true

module ML
  class Model
    attr_reader :x_data, :w_data, :b_data, :z_data

    def self.call(x_data, w_data, b_data)
      new(x_data, w_data, b_data).call
    end

    def initialize(x_data, w_data, b_data)
      @x_data = x_data
      @w_data = w_data
      @b_data = b_data
      @z_data = nil
    end

    def call
      @z_data = ML::Utils.dot(x_data, w_data).map { |row| [row[0] + b_data[0]] }

      z_data.map { |x| [1 / (1 + Math.exp(-x[0]))] }
    end
  end
end
